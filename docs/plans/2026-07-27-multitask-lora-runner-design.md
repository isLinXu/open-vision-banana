# Multi-Task LoRA Runner Design

## Goal

Provide a reproducible, auditable starting point for full-dataset multi-task
LoRA experiments without claiming that the original Open-Vision-Banana
generation path has been reproduced.

## Scope

The first supported smoke scope is five tasks whose paired RGB targets and
prompt paths are already loadable by `VisionTaskDataset`: depth, normal,
semantic segmentation, edge/structure, and generation. Depth remains a
relative-depth viridis task; it is not metric depth. Instance, referring,
keypoint, object detection, multiview, depth-edge, and specular-removal remain
audit-only until task-specific output/metric contracts are implemented.

The runner uses the existing SDXL/SSD LoRA instruction-tuning path with the
base VAE, UNet, and text encoders frozen. It samples task datasets with fixed
ratios, writes a manifest audit before loading the model, saves periodic LoRA
checkpoints, and evaluates each task on its declared validation subset using
task-appropriate sanity metrics. This is an exploratory multi-task generation
experiment, separate from the validated frozen-feature depth probe.

## Safety gates

- Every requested task must have non-empty train and validation declarations.
- Train and validation names must be disjoint per task.
- The runner must reject unsupported tasks rather than silently dropping them.
- A smoke run must finish with finite loss and a checkpoint containing the
  task list, ratios, split checksum, model, resolution, and seed.
- No promotion to a production/full model occurs without per-task validation
  improvement over the corresponding frozen/base control.

## Alternatives

- Reuse `train_v5_multitask.py`: rejected because it expects missing
  `v3_samples.jsonl` and an unverified ControlNet path.
- Start full multi-task training immediately: rejected because task contracts
  and validation gates are incomplete.
- Train all tasks with one generic depth prompt: rejected because prompt/task
  mismatch would confound the experiment.
