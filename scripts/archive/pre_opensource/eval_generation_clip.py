#!/usr/bin/env python3
"""Generation task CLIP score evaluation for SD3 LoRA.

Generates images from GenAI-Bench prompts, computes CLIP score (image-text similarity).
"""
import json, time, os, sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image

def main():
    # Paths
    model_path = Path.home() / "MyWork/models/sd3-medium-diffusers"
    lora_path = "output/sd3_checkpoints/lora_final.pt"
    gen_samples = Path("output/08662d03-9ca5-4394-b0c8-e019adb06a5d/paper_eval_samples/generation")
    out_dir = Path("output/sd3_eval_results/generation_images")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prompts
    with open(gen_samples / "labels.json") as f:
        samples = json.load(f)
    print(f"Loaded {len(samples)} generation prompts")

    # Load SD3 pipeline
    from diffusers import StableDiffusion3Pipeline
    print(f"Loading SD3 from {model_path}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(str(model_path), torch_dtype=torch.bfloat16)
    pipe.to("mps")
    pipe.set_progress_bar_config(disable=True)

    # Load and merge LoRA
    print(f"Loading LoRA from {lora_path}...")
    lora_state = torch.load(lora_path, map_location="cpu")
    # Count keys
    lora_keys = [k for k in lora_state if "lora_A" in k or "lora_B" in k]
    print(f"LoRA keys: {len(lora_keys)}")

    # Manual merge — keys use prefix "base_model.model." and param name without .weight suffix
    rank = 128
    alpha = 256
    scale = alpha / rank  # 2.0 for non-rslora
    merged = 0
    transformer = pipe.transformer
    with torch.no_grad():
        for name, param in transformer.named_parameters():
            # Remove .weight or .bias suffix to get module path
            module_path = name.rsplit(".", 1)[0] if name.endswith((".weight", ".bias")) else name
            key_a = f"base_model.model.{module_path}.lora_A.weight"
            key_b = f"base_model.model.{module_path}.lora_B.weight"
            if key_a in lora_state and key_b in lora_state:
                A = lora_state[key_a].to(param.dtype)
                B = lora_state[key_b].to(param.dtype)
                delta = (B @ A) * scale
                if delta.shape == param.shape:
                    param.add_(delta.to(param.device))
                    merged += 1
                merged += 1
    print(f"Merged {merged} LoRA layers")

    # Generate images
    results = []
    for i, sample in enumerate(samples):
        prompt = sample["prompt"]
        t0 = time.time()
        image = pipe(
            prompt=prompt,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=512,
            width=512,
        ).images[0]
        elapsed = time.time() - t0
        img_path = out_dir / f"gen_{i:03d}.png"
        image.save(img_path)
        results.append({"index": i, "prompt": prompt, "image_path": str(img_path), "time": elapsed})
        print(f"[{i+1}/{len(samples)}] {elapsed:.1f}s | {prompt[:60]}")

    # Compute CLIP score
    print("\nComputing CLIP scores...")
    try:
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        clip_model.eval()

        clip_scores = []
        with torch.no_grad():
            for r in results:
                image = Image.open(r["image_path"]).convert("RGB")
                inputs = clip_processor(text=[r["prompt"]], images=image, return_tensors="pt", padding=True)
                outputs = clip_model(**inputs)
                # CLIP score = 100 * cosine_similarity (image_embeds, text_embeds)
                score = torch.cosine_similarity(outputs.image_embeds, outputs.text_embeds).item()
                r["clip_score"] = score * 100
                clip_scores.append(r["clip_score"])
                print(f"  [{r['index']}] CLIP={r['clip_score']:.2f}")

        mean_clip = np.mean(clip_scores)
        std_clip = np.std(clip_scores)
        print(f"\nCLIP Score: {mean_clip:.2f} ± {std_clip:.2f}")
    except Exception as e:
        print(f"CLIP computation failed: {e}")
        mean_clip = 0.0
        std_clip = 0.0

    # Save results
    summary = {
        "n_samples": len(results),
        "clip_score_mean": float(mean_clip),
        "clip_score_std": float(std_clip),
    }
    with open("output/sd3_eval_results/generation_clip_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
    print(f"\nResults saved to output/sd3_eval_results/generation_clip_results.json")

if __name__ == "__main__":
    main()
