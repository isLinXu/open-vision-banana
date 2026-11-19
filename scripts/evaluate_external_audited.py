#!/usr/bin/env python3
"""Suite entry point: assert native generated canvas size before any resize.

Wraps the generic evaluator and intercepts the editing pipeline of ``--model-name``
(default qwen-image-2.1, i.e. QwenImage21Pipeline): every call must request the suite
resolution (``output_resolution`` for Qwen-Image-2.1, ``height``/``width`` for the other
editors) and return canvases of that size, and referring calls must carry query_v2 prompts.
"""
import functools
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

    def option(name):
        return argv[argv.index(name) + 1]

    resolution = int(option("--resolution"))
    referring = option("--benchmark") == "referring"
    if referring and option("--referring-prompt-version") != "query_v2":
        raise ValueError("This audited suite requires query_v2")
    prefix, suffix = BINARY_REFERRING_TEMPLATE.split("{query}")
    output = Path(option("--output-json"))
    model_name = option("--model-name") if "--model-name" in argv else "qwen-image-2.1"
    pipeline_class = getattr(diffusers, get_model_config(model_name).pipeline_class)
    sizes, prompt_count = [], 0
    original = pipeline_class.__call__

    @functools.wraps(original)
    def checked(pipeline, *positional, **kwargs):
        nonlocal prompt_count
        if "output_resolution" in kwargs:
            requested, wanted = kwargs["output_resolution"], resolution
        else:
            requested, wanted = (kwargs.get("height"), kwargs.get("width")), (resolution, resolution)
        if requested != wanted:
            raise ValueError(f"Wrong requested {model_name} pipeline resolution: {requested}")
        prompts = kwargs.get("prompt", positional[0] if positional else None)
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if referring and (not prompts or not all(isinstance(p, str) and p.startswith(prefix) and p.endswith(suffix) for p in prompts)):
            raise ValueError("Unexpected referring prompt at the pipeline")
        result = original(pipeline, *positional, **kwargs)
        for picture in result.images:
            if picture.size != (resolution, resolution):
                raise ValueError(f"Wrong raw pipeline output size: {picture.size}")
            sizes.append(picture.size)
        if referring:
            if len(prompts) != len(result.images):
                raise ValueError("Prompt/image count mismatch")
            prompt_count += len(prompts)
        return result

    pipeline_class.__call__ = checked
    try:
        entry = runpy.run_path(str(SOURCE / "scripts/evaluate_paper_benchmarks.py"))
        status = entry["main"]()
        if status:
            raise SystemExit(status)
        if not sizes:
            raise ValueError(f"No {model_name} pipeline outputs were verified")
        data = json.loads(output.read_text())
        data["metadata"]["resolution_verification"] = dict(raw_pipeline_output=[resolution, resolution], every_prediction_checked=True)
        if referring:
            data["metadata"]["referring_prompt_verification"] = dict(version="query_v2", every_prediction_checked=True)
        output.write_text(json.dumps(data, indent=2, allow_nan=False))
    finally:
        pipeline_class.__call__ = original
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".resolution-audit.json").write_text(json.dumps(dict(
            predictions_verified=len(sizes), raw_pipeline_sizes=sorted(set(sizes)),
            referring_prompts_verified=prompt_count), indent=2))


if __name__ == "__main__":
    main()
