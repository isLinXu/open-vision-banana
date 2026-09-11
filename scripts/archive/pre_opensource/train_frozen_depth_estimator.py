#!/usr/bin/env python3
"""Refit production frozen-feature depth heads from a passed resolution gate."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from compare_frozen_depth_features import (  # noqa: E402
    _clone_state_dict,
    _load_samples,
    _predict_in_batches,
    _resolve_device,
    _save_montages,
    _set_seed,
    aggregate_seed_metrics,
    build_batch_schedule,
    split_names_checksum,
    state_dict_checksum,
    tensor_checksum,
    write_json_report,
)

from open_vision_banana.core.depth_mapper import DepthRGBMapper  # noqa: E402
from open_vision_banana.data.depth_encoding import (  # noqa: E402
    audit_depth_target_encoding,
    relative_depth_prompt,
)
from open_vision_banana.depth_probe_checkpoint import (  # noqa: E402
    DepthProbeArtifact,
    load_depth_probe_artifact,
    save_depth_probe_artifact,
)
from open_vision_banana.inference.generator import BaseGenerator  # noqa: E402
from open_vision_banana.training.depth_probe import (  # noqa: E402
    DepthFeatureContract,
    FrozenSDXLDepthFeatureExtractor,
    ScalarDepthProbe,
    evaluate_scalar_depth_predictions,
    scalar_depth_probe_loss,
)
from open_vision_banana.training.tuner import VisionTaskDataset  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefitSelection:
    seed: int
    selected_step: int
    selection_train_count: int
    selection_batch_size: int
    initial_state_checksum: str
    source_artifact: DepthProbeArtifact


def declared_depth_names(data_dir: Path) -> tuple[list[str], list[str]]:
    manifest = data_dir / "train_val_split.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing depth split manifest: {manifest}")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    train_names = [str(name) for name in payload.get("train", {}).get("depth", [])]
    official_names = [str(name) for name in payload.get("val", {}).get("depth", [])]
    if not train_names or not official_names:
        raise ValueError("Depth manifest must contain non-empty train and val splits")
    if set(train_names).intersection(official_names):
        raise ValueError("Declared depth train and official validation splits overlap")
    return train_names, official_names


def load_refit_selections(report_path: Path) -> tuple[dict[str, Any], list[RefitSelection]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or report.get("gate_passed") is not True:
        raise ValueError("Selection report must be a passed frozen-feature gate")
    per_seed = report.get("per_seed")
    if not isinstance(per_seed, list) or not per_seed:
        raise ValueError("Selection report contains no per-seed results")
    selections = []
    for item in per_seed:
        seed = item.get("seed")
        selected_step = item.get("arms", {}).get("vae_ssd", {}).get("selected_step")
        checksum = item.get("matching", {}).get("initial_state_checksum")
        artifact_path = item.get("artifacts", {}).get("vae_ssd", {}).get("checkpoint")
        if (
            type(seed) is not int
            or type(selected_step) is not int
            or selected_step <= 0
            or not isinstance(checksum, str)
            or not checksum
            or not isinstance(artifact_path, str)
        ):
            raise ValueError("Selection report contains an invalid VAE+SSD seed record")
        artifact = load_depth_probe_artifact(artifact_path)
        if artifact.selected_step != selected_step or artifact.training.get("seed") != seed:
            raise ValueError(f"Selection artifact metadata mismatch for seed {seed}")
        selection_names = artifact.splits.get("train_fit_names")
        selection_batch_size = artifact.training.get("batch_size")
        if (
            not isinstance(selection_names, list)
            or not selection_names
            or type(selection_batch_size) is not int
            or selection_batch_size <= 0
        ):
            raise ValueError(f"Selection artifact lacks exposure metadata for seed {seed}")
        selections.append(
            RefitSelection(
                seed=seed,
                selected_step=selected_step,
                selection_train_count=len(selection_names),
                selection_batch_size=selection_batch_size,
                initial_state_checksum=checksum,
                source_artifact=artifact,
            )
        )
    if len({selection.seed for selection in selections}) != len(selections):
        raise ValueError("Selection report contains duplicate seeds")
    reference = selections[0].source_artifact
    for selection in selections[1:]:
        artifact = selection.source_artifact
        comparable = (
            artifact.model,
            artifact.prompt,
            artifact.colormap,
            artifact.resolution,
            artifact.timestep,
            artifact.width,
            artifact.upsample_stages,
            artifact.feature_contract,
            artifact.feature_mask,
        )
        expected = (
            reference.model,
            reference.prompt,
            reference.colormap,
            reference.resolution,
            reference.timestep,
            reference.width,
            reference.upsample_stages,
            reference.feature_contract,
            reference.feature_mask,
        )
        if comparable != expected:
            raise ValueError("Selected seed artifacts do not share one inference contract")
    return report, selections


def scale_refit_steps(
    *,
    selected_steps: int,
    selection_train_count: int,
    selection_batch_size: int,
    refit_train_count: int,
    refit_batch_size: int,
) -> int:
    """Preserve the dev-selected number of sample exposures across data sizes."""
    values = (
        selected_steps,
        selection_train_count,
        selection_batch_size,
        refit_train_count,
        refit_batch_size,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("step scaling inputs must be positive integers")
    exposure_ratio = selected_steps * selection_batch_size / selection_train_count
    return max(1, round(exposure_ratio * refit_train_count / refit_batch_size))


def _cache_declared_features(
    generator: BaseGenerator,
    data_dir: Path,
    *,
    names: list[str],
    resolution: int,
    prompt: str,
    timestep: int,
    mapper: DepthRGBMapper,
    storage_dtype: torch.dtype,
) -> tuple[DepthFeatureContract, tuple[torch.Tensor, ...], torch.Tensor, list[list[int]]]:
    dataset = VisionTaskDataset(
        str(data_dir / "depth"),
        resolution=resolution,
        task="depth",
    )
    by_name = {
        Path(sample["image_path"]).name: index
        for index, sample in enumerate(dataset.samples)
    }
    missing = [name for name in names if name not in by_name]
    if missing:
        raise FileNotFoundError(f"Declared depth samples do not resolve: {missing[:8]}")
    extractor = FrozenSDXLDepthFeatureExtractor(generator, timestep=timestep)
    contract: DepthFeatureContract | None = None
    levels: list[list[torch.Tensor]] | None = None
    targets: list[torch.Tensor] = []
    for position, name in enumerate(names, start=1):
        sample = dataset[by_name[name]]
        image = sample["image"].unsqueeze(0)
        sample_contract, sample_features = extractor.extract(image, prompt=prompt)
        if contract is None:
            contract = sample_contract
            levels = [[] for _ in sample_features]
        elif sample_contract != contract:
            raise RuntimeError("Frozen feature contract changed between samples")
        assert levels is not None
        for feature_name, destination, feature in zip(
            sample_contract.names,
            levels,
            sample_features,
            strict=True,
        ):
            stored = feature.to(device="cpu", dtype=storage_dtype)
            if not torch.isfinite(stored).all():
                raise FloatingPointError(
                    f"Feature {feature_name} for {name} is non-finite after {storage_dtype} conversion"
                )
            destination.append(stored)
        target = mapper.rgb_to_normalized(sample["target_rgb"].unsqueeze(0))
        target = target.to(device="cpu", dtype=storage_dtype)
        if not torch.isfinite(target).all():
            raise FloatingPointError(f"Depth target for {name} is non-finite")
        targets.append(target)
        if position % 32 == 0 or position == len(names):
            logger.info("Cached frozen features for %d/%d images", position, len(names))
    if contract is None or levels is None:
        raise RuntimeError("No frozen features were extracted")
    features = tuple(torch.cat(level, dim=0) for level in levels)
    target_tensor = torch.cat(targets, dim=0)
    shapes = [list(feature.shape) for feature in features]
    return contract, features, target_tensor, shapes


@dataclass
class RefitResult:
    state_dict: dict[str, torch.Tensor]
    train_sample_metrics: dict[str, float]
    official_val_metrics: dict[str, float]
    official_val_prediction: torch.Tensor
    runtime_seconds: float
    schedule_checksum: str
    trainable_parameters: int


def _train_fixed_head(
    *,
    selection: RefitSelection,
    contract: DepthFeatureContract,
    train_features: tuple[torch.Tensor, ...],
    official_features: tuple[torch.Tensor, ...],
    train_target: torch.Tensor,
    official_target: torch.Tensor,
    feature_mask: tuple[bool, ...],
    refit_steps: int,
    batch_size: int,
    train_eval_samples: int,
    device: torch.device,
) -> RefitResult:
    artifact = selection.source_artifact
    learning_rate = float(artifact.training["learning_rate"])
    gradient_weight = float(artifact.training["gradient_weight"])
    pearson_weight = float(artifact.training["pearson_weight"])
    _set_seed(selection.seed)
    started = time.perf_counter()
    head = ScalarDepthProbe(
        contract,
        width=artifact.width,
        upsample_stages=artifact.upsample_stages,
    )
    initial_checksum = state_dict_checksum(_clone_state_dict(head))
    if initial_checksum != selection.initial_state_checksum:
        raise RuntimeError(
            f"Seed {selection.seed} initial state checksum differs from selection gate"
        )
    head = head.to(device=device, dtype=torch.float32)
    trainable_parameters = sum(parameter.numel() for parameter in head.parameters())
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=refit_steps,
        eta_min=learning_rate * 0.01,
    )
    schedule = build_batch_schedule(
        num_samples=train_target.shape[0],
        batch_size=batch_size,
        steps=refit_steps,
        seed=selection.seed + 1000,
    )
    schedule_checksum = tensor_checksum(schedule)
    output_size = tuple(int(value) for value in train_target.shape[-2:])
    for step, indices in enumerate(schedule, start=1):
        prediction = head(
            tuple(feature[indices] for feature in train_features),
            output_size=output_size,
            feature_mask=feature_mask,
        )
        target_batch = train_target[indices].to(device=device, dtype=torch.float32)
        loss = scalar_depth_probe_loss(
            prediction,
            target_batch,
            gradient_weight=gradient_weight,
            pearson_weight=pearson_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % 250 == 0 or step == refit_steps:
            logger.info(
                "seed %d exposure-scaled refit step %d/%d | loss=%.5f",
                selection.seed,
                step,
                refit_steps,
                float(loss.detach().cpu()),
            )
    head.eval()
    train_count = min(train_eval_samples, train_target.shape[0])
    train_prediction = _predict_in_batches(
        head,
        tuple(feature[:train_count] for feature in train_features),
        feature_mask=feature_mask,
        output_size=output_size,
        batch_size=batch_size,
    )
    official_prediction = _predict_in_batches(
        head,
        official_features,
        feature_mask=feature_mask,
        output_size=output_size,
        batch_size=batch_size,
    )
    return RefitResult(
        state_dict=_clone_state_dict(head),
        train_sample_metrics=evaluate_scalar_depth_predictions(
            train_prediction,
            train_target[:train_count],
        ),
        official_val_metrics=evaluate_scalar_depth_predictions(
            official_prediction,
            official_target,
        ),
        official_val_prediction=official_prediction,
        runtime_seconds=time.perf_counter() - started,
        schedule_checksum=schedule_checksum,
        trainable_parameters=trainable_parameters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-report", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--train-eval-samples", type=int, default=128)
    parser.add_argument("--visualization-count", type=int, default=4)
    parser.add_argument(
        "--feature-storage-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if min(args.batch_size, args.train_eval_samples, args.visualization_count) <= 0:
        parser.error("batch-size, train-eval-samples, and visualization-count must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    started = time.perf_counter()
    selection_report = args.selection_report.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report, selections = load_refit_selections(selection_report)
    reference = selections[0].source_artifact
    train_names, official_names = declared_depth_names(data_dir)
    split_checksum = split_names_checksum(train_names, [], official_names)
    audit = audit_depth_target_encoding(
        data_dir / "depth" / "targets",
        allowed_names=set([*train_names, *official_names]),
    )
    if audit.detected_colormap != reference.colormap:
        raise ValueError(
            f"Dataset colormap {audit.detected_colormap!r} does not match gate {reference.colormap!r}"
        )
    if relative_depth_prompt(audit.detected_colormap) != reference.prompt:
        raise ValueError("Dataset prompt contract does not match the selection gate")
    mapper = DepthRGBMapper(reference.colormap)
    device = _resolve_device(args.device)
    storage_dtype = getattr(torch, args.feature_storage_dtype)
    all_names = [*train_names, *official_names]
    logger.info(
        "Production refit: train=%d official_val=%d resolution=%d seeds=%s device=%s",
        len(train_names),
        len(official_names),
        reference.resolution,
        [selection.seed for selection in selections],
        device,
    )
    generator = BaseGenerator(
        model_name=reference.model,
        device=str(device),
        dtype=torch.bfloat16 if device.type in {"mps", "cuda"} else torch.float32,
    )
    contract, all_features, all_target, feature_shapes = _cache_declared_features(
        generator,
        data_dir,
        names=all_names,
        resolution=reference.resolution,
        prompt=reference.prompt,
        timestep=reference.timestep,
        mapper=mapper,
        storage_dtype=storage_dtype,
    )
    del generator
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    expected_contract = DepthFeatureContract.from_dict(reference.feature_contract)
    if contract != expected_contract:
        raise ValueError("Full-data feature contract differs from the selection gate")
    train_count = len(train_names)
    train_features = tuple(feature[:train_count] for feature in all_features)
    official_features = tuple(feature[train_count:] for feature in all_features)
    train_target = all_target[:train_count]
    official_target = all_target[train_count:]
    visualization_count = min(args.visualization_count, len(official_names))
    visualization_images, visualization_rgb = _load_samples(
        data_dir,
        names=official_names[:visualization_count],
        resolution=reference.resolution,
    )

    seed_reports = []
    official_metrics = []
    for selection in selections:
        refit_steps = scale_refit_steps(
            selected_steps=selection.selected_step,
            selection_train_count=selection.selection_train_count,
            selection_batch_size=selection.selection_batch_size,
            refit_train_count=len(train_names),
            refit_batch_size=args.batch_size,
        )
        logger.info(
            "seed %d selection step %d on %d samples -> %d full-data refit steps",
            selection.seed,
            selection.selected_step,
            selection.selection_train_count,
            refit_steps,
        )
        result = _train_fixed_head(
            selection=selection,
            contract=contract,
            train_features=train_features,
            official_features=official_features,
            train_target=train_target,
            official_target=official_target,
            feature_mask=reference.feature_mask,
            refit_steps=refit_steps,
            batch_size=args.batch_size,
            train_eval_samples=args.train_eval_samples,
            device=device,
        )
        artifact_dir = save_depth_probe_artifact(
            output_dir / f"seed{selection.seed}_vae_ssd_final",
            head_state_dict=result.state_dict,
            feature_contract=contract.to_dict(),
            feature_mask=reference.feature_mask,
            model=reference.model,
            prompt=reference.prompt,
            colormap=reference.colormap,
            resolution=reference.resolution,
            timestep=reference.timestep,
            width=reference.width,
            upsample_stages=reference.upsample_stages,
            selected_step=refit_steps,
            metrics={
                "train_sample": result.train_sample_metrics,
                "official_val": result.official_val_metrics,
            },
            splits={
                "train_names": train_names,
                "official_val_names": official_names,
                "checksum": split_checksum,
            },
            training={
                "mode": "dev-selected-exposure-full-train-refit",
                "seed": selection.seed,
                "steps": refit_steps,
                "selection_steps": selection.selected_step,
                "selection_train_count": selection.selection_train_count,
                "selection_batch_size": selection.selection_batch_size,
                "batch_size": args.batch_size,
                "learning_rate": reference.training["learning_rate"],
                "gradient_weight": reference.training["gradient_weight"],
                "pearson_weight": reference.training["pearson_weight"],
                "feature_storage_dtype": args.feature_storage_dtype,
                "selection_report": str(selection_report),
                "selection_initial_state_checksum": selection.initial_state_checksum,
                "batch_schedule_checksum": result.schedule_checksum,
            },
        )
        visualization_dir = output_dir / "visualizations"
        visualization_dir.mkdir(parents=True, exist_ok=True)
        visualization_paths = _save_montages(
            visualization_dir,
            arm_name=f"seed{selection.seed}_vae_ssd_final",
            names=official_names[:visualization_count],
            images=visualization_images,
            target_rgb=visualization_rgb,
            prediction=result.official_val_prediction[:visualization_count],
            colormap=reference.colormap,
            limit=visualization_count,
        )
        official_metrics.append(result.official_val_metrics)
        seed_reports.append(
            {
                "seed": selection.seed,
                "selection_step": selection.selected_step,
                "refit_steps": refit_steps,
                "checkpoint": str(artifact_dir),
                "visualizations": visualization_paths,
                "train_sample_metrics": result.train_sample_metrics,
                "official_val_metrics": result.official_val_metrics,
                "runtime_seconds": result.runtime_seconds,
                "schedule_checksum": result.schedule_checksum,
                "trainable_parameters": result.trainable_parameters,
            }
        )
    final_report = {
        "stage": "production_refit",
        "decision": "frozen_depth_estimator_ready",
        "selection_report": str(selection_report),
        "selection_gate": {
            "resolution": report["resolution"],
            "gate_passed": report["gate_passed"],
            "split_checksum": report["splits"]["checksum"],
        },
        "model": reference.model,
        "data_dir": str(data_dir),
        "resolution": reference.resolution,
        "prompt": reference.prompt,
        "colormap": reference.colormap,
        "feature_contract": contract.to_dict(),
        "feature_shapes": feature_shapes,
        "feature_storage_dtype": args.feature_storage_dtype,
        "splits": {
            "train_count": len(train_names),
            "official_val_count": len(official_names),
            "checksum": split_checksum,
        },
        "aggregate_official_val": aggregate_seed_metrics(official_metrics),
        "per_seed": seed_reports,
        "runtime_seconds": time.perf_counter() - started,
    }
    write_json_report(output_dir / "result.json", final_report)
    logger.info(
        "Production refit official-val aggregate: %s",
        json.dumps(final_report["aggregate_official_val"], sort_keys=True),
    )


if __name__ == "__main__":
    main()
