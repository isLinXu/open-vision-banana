#!/usr/bin/env python3
"""Qwen suite entry point: assert native generated canvas size before any resize.

The generic evaluator remains usable with other model families. This wrapper is
deliberately QwenImage21-specific and verifies query_v2 prompts at the pipeline.
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
    from open_vision_banana.referring_contract import BINARY_REFERRING_TEMPLATE

    def option(name):
        return argv[argv.index(name) + 1]

    resolution = int(option("--resolution"))
    referring = option("--benchmark") == "referring"
    if referring and option("--referring-prompt-version") != "query_v2":
        raise ValueError("This audited suite requires query_v2")
    prefix, suffix = BINARY_REFERRING_TEMPLATE.split("{query}")
    output = Path(option("--output-json"))
    sizes, prompt_count = [], 0
    original = diffusers.QwenImage21Pipeline.__call__

    @functools.wraps(original)
    def checked(pipeline, *positional, **kwargs):
        nonlocal prompt_count
        if kwargs.get("output_resolution") != resolution:
            raise ValueError("Wrong requested Qwen pipeline resolution")
        prompts = kwargs.get("prompt", positional[0] if positional else None)
        prompts = [prompts] if isinstance(prompts, str) else prompts
        if referring and (not prompts or not all(isinstance(p, str) and p.startswith(prefix) and p.endswith(suffix) for p in prompts)):
            raise ValueError("Unexpected referring prompt at Qwen pipeline")
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

    diffusers.QwenImage21Pipeline.__call__ = checked
    try:
        entry = runpy.run_path(str(SOURCE / "scripts/evaluate_paper_benchmarks.py"))
        status = entry["main"]()
        if status:
            raise SystemExit(status)
        if not sizes:
            raise ValueError("No Qwen outputs were verified")
        data = json.loads(output.read_text())
        data["metadata"]["resolution_verification"] = dict(raw_pipeline_output=[resolution, resolution], every_prediction_checked=True)
        if referring:
            data["metadata"]["referring_prompt_verification"] = dict(version="query_v2", every_prediction_checked=True)
        output.write_text(json.dumps(data, indent=2, allow_nan=False))
    finally:
        diffusers.QwenImage21Pipeline.__call__ = original
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_suffix(".resolution-audit.json").write_text(json.dumps(dict(
            predictions_verified=len(sizes), raw_pipeline_sizes=sorted(set(sizes)),
            referring_prompts_verified=prompt_count), indent=2))


if __name__ == "__main__":
    main()
