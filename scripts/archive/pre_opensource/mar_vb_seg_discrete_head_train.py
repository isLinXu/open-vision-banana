#!/usr/bin/env python3
"""Train/evaluate a categorical head from cached MAR token features."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.utils import save_image

NUM_CLASSES = 150
IGNORE_LABEL = 255


class TokenClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim),
            nn.GELU(), nn.Linear(hidden_dim, NUM_CLASSES),
        )

    def forward(self, features):
        return self.layers(features)


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--palette", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-samples", type=int, default=64)
    p.add_argument("--val-samples", type=int, default=32)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--best-by", choices=("train", "val"), default="val")
    return p.parse_args()


def rows(manifest: Path, split: str, limit: int):
    all_rows = [json.loads(line) for line in manifest.open() if line.strip()]
    selected = [r for r in all_rows if r.get("task") == "seg" and r.get("split") == split]
    selected.sort(key=lambda r: r["input"])
    if limit > len(selected):
        raise ValueError(f"need {limit} {split} rows, found {len(selected)}")
    indices = np.linspace(0, len(selected) - 1, num=limit, dtype=np.int64)
    return [selected[int(index)] for index in indices]


def palette_lut(path: Path):
    data = json.loads(path.read_text())
    palette = np.asarray(data["class_palette"], dtype=np.uint8)
    lut = np.full(1 << 24, IGNORE_LABEL, dtype=np.uint8)
    packed = (palette[:, 0].astype(np.uint32) << 16) | (palette[:, 1].astype(np.uint32) << 8) | palette[:, 2]
    lut[packed] = np.arange(NUM_CLASSES, dtype=np.uint8)
    return palette, lut


def labels(path: str, size: int, lut: np.ndarray):
    with Image.open(path) as image:
        arr = np.asarray(image.convert("RGB").resize((size, size), Image.NEAREST), dtype=np.uint8)
    packed = (arr[..., 0].astype(np.uint32) << 16) | (arr[..., 1].astype(np.uint32) << 8) | arr[..., 2]
    return lut[packed]


def grid_labels(full: np.ndarray, grid: int):
    stride = full.shape[0] // grid
    blocks = full.reshape(grid, stride, grid, stride).transpose(0, 2, 1, 3).reshape(grid * grid, -1)
    out = np.full(len(blocks), IGNORE_LABEL, dtype=np.uint8)
    for i, block in enumerate(blocks):
        valid = block[block != IGNORE_LABEL]
        if len(valid):
            out[i] = np.bincount(valid, minlength=NUM_CLASSES).argmax()
    return out


def load_split(cache: Path, split: str, selected, image_size: int, lut):
    features = torch.stack([torch.load(cache / split / f"{i:05d}.pt", map_location="cpu", weights_only=True).float()
                            for i in range(len(selected))])
    full = np.stack([labels(row["target"], image_size, lut) for row in selected])
    grid = np.stack([grid_labels(x, int(round(features.shape[1] ** 0.5))) for x in full])
    inputs = []
    for row in selected:
        with Image.open(row["input"]) as image:
            arr = np.asarray(image.convert("RGB").resize((image_size, image_size), Image.BICUBIC), dtype=np.float32) / 127.5 - 1
        inputs.append(torch.from_numpy(arr).permute(2, 0, 1))
    return features, grid, full, torch.stack(inputs)


def metrics(pred, target):
    valid = target.reshape(-1) != IGNORE_LABEL
    cm = np.bincount(NUM_CLASSES * target.reshape(-1)[valid].astype(np.int64) + pred.reshape(-1)[valid].astype(np.int64), minlength=NUM_CLASSES**2).reshape(NUM_CLASSES, NUM_CLASSES)
    tp = np.diag(cm).astype(float); tt = cm.sum(1).astype(float); pt = cm.sum(0).astype(float)
    union = tt + pt - tp; present = union > 0
    return {"pixel_accuracy": float(tp.sum() / max(1, tt.sum())), "miou": float((tp[present] / union[present]).mean()), "present_classes": int(present.sum()), "valid_pixels": int(tt.sum())}


def expand(grid_pred, grid, size):
    stride = size // grid
    return np.repeat(np.repeat(grid_pred.reshape(-1, grid, grid), stride, 1), stride, 2)


def panel(path, inputs, target, pred, palette):
    def rgb(labels):
        out = np.zeros((*labels.shape, 3), dtype=np.uint8); valid = labels != IGNORE_LABEL; out[valid] = palette[labels[valid]]
        return torch.from_numpy(out).permute(0, 3, 1, 2).float() / 255
    src = (inputs.clamp(-1, 1) + 1) / 2
    images = torch.stack((src, rgb(target), rgb(pred)), 1).reshape(-1, 3, target.shape[-2], target.shape[-1])
    save_image(images, path, nrow=3)


def main():
    a = args(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    a.output.mkdir(parents=True, exist_ok=True)
    palette, lut = palette_lut(a.palette)
    train_rows = rows(a.manifest, "train", a.train_samples); val_rows = rows(a.manifest, "val", a.val_samples)
    train_x, train_y, train_full, train_in = load_split(a.cache, "train", train_rows, a.image_size, lut)
    val_x, val_y, val_full, val_in = load_split(a.cache, "val", val_rows, a.image_size, lut)
    grid = int(round(train_x.shape[1] ** 0.5)); feature_dim = train_x.shape[2]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    head = TokenClassifier(feature_dim, a.hidden_dim).to(device)
    tx = train_x.to(device); ty = torch.from_numpy(train_y.astype(np.int64)).to(device)
    valid = ty != IGNORE_LABEL
    freq = torch.bincount(ty[valid].reshape(-1), minlength=NUM_CLASSES).float()
    weights = torch.zeros(NUM_CLASSES, device=device); active = freq > 0; weights[active] = freq[active].rsqrt(); weights[active] /= weights[active].mean()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0)
    best_state = None; best_value = -1.; history = []; batch = min(a.batch_size, a.train_samples)
    for step in range(1, a.steps + 1):
        idx = torch.randint(a.train_samples, (batch,), device=device)
        logits = head(tx[idx]); target = ty[idx]; mask = target != IGNORE_LABEL
        loss = F.cross_entropy(logits[mask], target[mask], weight=weights)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % max(1, min(100, a.steps // 10)) == 0 or step == a.steps:
            with torch.no_grad():
                train_pred = head(tx).argmax(-1).cpu().numpy().astype(np.uint8)
                val_pred = head(val_x.to(device)).argmax(-1).cpu().numpy().astype(np.uint8)
            tm = metrics(train_pred, train_y); vm = metrics(val_pred, val_y); history.append({"step":step,"loss":float(loss.detach()),"train_grid":tm,"val_grid":vm})
            chosen = tm if a.best_by == "train" else vm
            if chosen["miou"] > best_value:
                best_value = chosen["miou"]; best_state = {k:v.detach().cpu().clone() for k,v in head.state_dict().items()}; best_step = step
            print(f"[head] step={step} loss={float(loss.detach()):.5f} train_miou={tm['miou']:.4f} val_miou={vm['miou']:.4f}", flush=True)
    head.load_state_dict(best_state); torch.save({"head":best_state,"step":best_step,"best_by":a.best_by,"best_miou":best_value}, a.output / "classifier_head.pt")
    with torch.no_grad():
        train_pred = head(tx).argmax(-1).cpu().numpy().astype(np.uint8); val_pred = head(val_x.to(device)).argmax(-1).cpu().numpy().astype(np.uint8)
    train_exp = expand(train_pred, grid, a.image_size); val_exp = expand(val_pred, grid, a.image_size)
    train_oracle = expand(train_y, grid, a.image_size); val_oracle = expand(val_y, grid, a.image_size)
    result = {"train":{"grid":metrics(train_pred,train_y),"expanded_pixel":metrics(train_exp,train_full),"expanded_oracle_pixel":metrics(train_oracle,train_full)},"val":{"grid":metrics(val_pred,val_y),"expanded_pixel":metrics(val_exp,val_full),"expanded_oracle_pixel":metrics(val_oracle,val_full)}}
    panel(a.output/"panel_train.png",train_in,train_full,train_exp,palette); panel(a.output/"panel_val.png",val_in,val_full,val_exp,palette)
    report={"probe":"cached_frozen_mar_conditional_features_direct_150_class_head","manifest":str(a.manifest.resolve()),"cache":str(a.cache.resolve()),"palette":str(a.palette.resolve()),"grid_size":grid,"image_size":a.image_size,"train_samples":a.train_samples,"val_samples":a.val_samples,"steps":a.steps,"batch_size":batch,"lr":a.lr,"hidden_dim":a.hidden_dim,"best_by":a.best_by,"best_step":best_step,"best_miou":best_value,"history":history,"results":result}
    (a.output/"metrics.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__": main()
