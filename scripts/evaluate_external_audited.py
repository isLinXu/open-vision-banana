#!/usr/bin/env python3
"""Suite entry point: assert native generated canvas size before any resize.

Wraps the generic evaluator and intercepts the editing pipeline of ``--model-name``
(default qwen-image-2.1, i.e. QwenImage21Pipeline): every call must request the suite
resolution (``output_resolution`` for Qwen-Image-2.1, ``height``/``width`` for the other
editors) and return canvases of that size, and referring calls must carry query_v2 prompts.

A backbone whose predictions never reach a diffusers pipeline ``__call__`` is intercepted at
the generator method that produces them instead (``generator_audited`` below), so the same
guarantee holds for it: every prediction is checked, and an interception that caught nothing
still fails the run because no sizes were recorded.
"""
import functools
import inspect
import json
from pathlib import Path
import runpy
import sys

SOURCE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SOURCE / "src"))


def main():
    argv = sys.argv[1:]
    # Preflight never imports or loads a diffusion model.
    if "--preflight-only" in argv:
        runpy.run_path(str(SOURCE / "scripts/evaluate_paper_benchmarks.py"), run_name="__main__")
        return
    import diffusers
    from open_vision_banana.model_registry import get_model_config
    from open_vision_banana.referring_contract import BINARY_REFERRING_TEMPLATE

    def option(name, default=None):
        return argv[argv.index(name) + 1] if name in argv else default

    resolution = int(option("--resolution"))
    referring = option("--benchmark") == "referring"
    if referring and option("--referring-prompt-version") != "query_v2":
        raise ValueError("This audited suite requires query_v2")
    prefix, suffix = BINARY_REFERRING_TEMPLATE.split("{query}")
    output = Path(option("--output-json"))
    model_name = option("--model-name", "qwen-image-2.1")
    class_name = get_model_config(model_name).pipeline_class
    sizes, prompt_count = [], 0
    patched = []  # (owner, attribute, original), put back in the finally below

    def check_prompts(prompts):
        if referring and (not prompts or not all(isinstance(p, str) and p.startswith(prefix) and p.endswith(suffix) for p in prompts)):
            raise ValueError("Unexpected referring prompt at the pipeline")

    def tally_prompts(prompts, images):
        nonlocal prompt_count
        if referring:
            if len(prompts) != images:
                raise ValueError("Prompt/image count mismatch")
            prompt_count += len(prompts)

    # Backbones whose predictions never pass through a diffusers pipeline __call__, mapped to the
    # generator method that produces them instead. MOSS-Image-Base has no reference-image input, so
    # the generator denoises the right half of an in-context panel by driving the transformer,
    # scheduler and VAE itself: patching its pipeline class would verify nothing, and that class is
    # remote code in the model repo rather than an attribute of diffusers.
    generator_audited = {"MOSSImagePipeline": "_generate_moss_panel"}

    if class_name in generator_audited:
        from open_vision_banana.inference.generator import BaseGenerator

        method = generator_audited[class_name]
        original = getattr(BaseGenerator, method)
        raw_output = [resolution, resolution]

        @functools.wraps(original)
        def checked_generator(self, *positional, **kwargs):
            prompts = kwargs.get("prompt", positional[0] if positional else None)
            prompts = [prompts] if isinstance(prompts, str) else list(prompts)
            check_prompts(prompts)
            result = original(self, *positional, **kwargs)
            # [B, C, H, W] in [0, 1]. The suite's rasters are already a multiple of 16, so the
            # decoded panel half is the suite resolution with nothing resized on the way out.
            pictures = result if result.dim() == 4 else result.unsqueeze(0)
            for picture in pictures:
                size = (picture.shape[-1], picture.shape[-2])
                if size != (resolution, resolution):
                    raise ValueError(f"Wrong raw {model_name} output size: {size}")
                sizes.append(size)
            tally_prompts(prompts, len(pictures))
            return result

        setattr(BaseGenerator, method, checked_generator)
        patched.append((BaseGenerator, method, original))
    else:
        pipeline_class = getattr(diffusers, class_name, None)
        if pipeline_class is None:
            raise ValueError(f"{model_name}: {class_name} is not a diffusers pipeline the audit can intercept")
        # The canvas argument is the pipeline's own: Qwen-Image-2.1 takes output_resolution, the others height/width
        parameters = inspect.signature(pipeline_class.__call__).parameters
        by_side = "output_resolution" in parameters
        # LongCat-Image-Edit takes no canvas argument: it always renders the source's aspect at a 1024**2
        # area, so only the native output is recorded and the generator resizes it, as in inference
        native = not by_side and not {"height", "width"} <= set(parameters)
        original = pipeline_class.__call__
        raw_output = "pipeline_native" if native else [resolution, resolution]

        @functools.wraps(original)
        def checked(pipeline, *positional, **kwargs):
            if by_side:
                requested, wanted = kwargs.get("output_resolution"), resolution
            else:
                requested, wanted = (kwargs.get("height"), kwargs.get("width")), (resolution, resolution)
            if not native and requested != wanted:
                raise ValueError(f"Wrong requested {model_name} pipeline resolution: {requested}")
            prompts = kwargs.get("prompt", positional[0] if positional else None)
            prompts = [prompts] if isinstance(prompts, str) else prompts
            check_prompts(prompts)
            result = original(pipeline, *positional, **kwargs)
            for picture in result.images:
                if not native and picture.size != (resolution, resolution):
                    raise ValueError(f"Wrong raw pipeline output size: {picture.size}")
                sizes.append(picture.size)
            tally_prompts(prompts, len(result.images))
            return result

        pipeline_class.__call__ = checked
        patched.append((pipeline_class, "__call__", original))
    try:
        entry = runpy.run_path(str(SOURCE / "scripts/evaluate_paper_benchmarks.py"))
        status = entry["main"]()
        if status:
            raise SystemExit(status)
        if not sizes:
            raise ValueError(f"No {model_name} pipeline outputs were verified")
        data = json.loads(output.read_text())
        data["metadata"]["resolution_verification"] = dict(raw_pipeline_output=raw_output, every_prediction_checked=True)
        if referring:
            data["metadata"]["referring_prompt_verification"] = dict(version="query_v2", every_prediction_checked=True)
        output.write_text(json.dumps(data, indent=2, allow_nan=False))
    finally:
        for owner, attribute, restored in patched:
            setattr(owner, attribute, restored)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".resolution-audit.json").write_text(json.dumps(dict(
            predictions_verified=len(sizes), raw_pipeline_sizes=sorted(set(sizes)),
            referring_prompts_verified=prompt_count), indent=2))


if __name__ == "__main__":
    main()
