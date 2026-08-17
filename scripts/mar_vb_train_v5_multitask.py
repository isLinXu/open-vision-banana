#!/usr/bin/env python3
"""MAR-VB v5_full Multi-Task: depth + normal + segmentation + structure on MAR.

Multi-task instruction tuning: model conditioned on task label predicts
task-specific target from input image.

Tasks & labels:
  depth=0    normal=1    segmentation=2    structure=3

Usage:
  VB_STEPS=3000 VB_TASKS=depth,normal,seg,struct python scripts/mar_vb_train_v5_multitask.py
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.utils import save_image

if not torch.cuda.is_available() and torch.backends.mps.is_available():
    torch.Tensor.cuda = lambda self, *a, **k: self.to("mps")

sys.path.insert(0, "/Users/gatilin/MyWork/models/ckpts")
from mar.mar import mar_base, mar_large, mask_by_order
from mar.vae import AutoencoderKL

# ── Config ────────────────────────────────────────────────────────────────
CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
DATA_ROOT = os.environ.get("VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("VB_OUT",
    str(PROJECT_ROOT / "output" / "mar_vb_v5_multitask")))

IMG_SIZE = int(os.environ.get("VB_IMG_SIZE", "384"))
N_TRAIN = int(os.environ.get("VB_N_TRAIN", "20000"))
N_VAL = int(os.environ.get("VB_VAL", "640"))  # 16 per task * 4 tasks * 10x safety
N_VAL_PER_TASK = int(os.environ.get("VB_N_VAL_PER_TASK", "16"))  # val samples per task
STEPS = int(os.environ.get("VB_STEPS", "3000"))
TOTAL_STEPS = int(os.environ.get("VB_TOTAL_STEPS", str(STEPS)))
BATCH = int(os.environ.get("VB_BATCH", "1"))
NUM_WORKERS = int(os.environ.get("VB_NUM_WORKERS", "2"))
SAVE_LATEST_EVERY = int(os.environ.get("VB_SAVE_LATEST_EVERY", "20"))
# Staged training: train at one img_size, then resize pos and continue at another
STAGE_A_SIZE = int(os.environ.get("VB_STAGE_A_SIZE", "0"))  # 0 = disabled
STAGE_A_STEPS = int(os.environ.get("VB_STAGE_A_STEPS", "0"))  # steps to run at stage A
STAGE_A_BATCH = int(os.environ.get("VB_STAGE_A_BATCH", "2"))
LR = float(os.environ.get("VB_LR", "1e-5"))
WARMUP = int(os.environ.get("VB_WARMUP", "50"))
WEIGHT_DECAY = float(os.environ.get("VB_WD", "0.08"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
DROPOUT = float(os.environ.get("VB_DROPOUT", "0.2"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "100"))
SAMPLE_EVERY = int(os.environ.get("VB_SAMPLE_EVERY", "500"))
# EMA for stable fine-tuning
EMA_DECAY = float(os.environ.get("VB_EMA", "0.999"))
SAMPLE_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
SAMPLE_SKIP = os.environ.get("VB_SAMPLE_SKIP", "0") == "1"
SKIP_FINAL_SAVE = os.environ.get("VB_SKIP_FINAL_SAVE", "0") == "1"
RESUME = os.environ.get("VB_RESUME", "")
RESET_EMA = os.environ.get("VB_RESET_EMA", "0") == "1"  # skip loading EMA shadow from ckpt
RESET_OPT = os.environ.get("VB_RESET_OPT", "0") == "1"  # fresh Adam moments on distribution changes
RESET_BEST = os.environ.get("VB_RESET_BEST", "0") == "1"  # new validation population/scale
PAIR_MANIFEST = os.environ.get("VB_PAIR_MANIFEST", "")
SEED = 42
VAE_SCALE = 0.2325
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Task weights: higher = more samples in batch. Default equal.
TASK_WEIGHTS = {
    "depth": float(os.environ.get("VB_W_DEPTH", "2.0")),
    "normal": float(os.environ.get("VB_W_NORMAL", "1.0")),
    "seg": float(os.environ.get("VB_W_SEG", "3.0")),
    "structure": float(os.environ.get("VB_W_STRUCT", "2.0")),
}
TASK_LABEL = {"depth": 0, "normal": 1, "seg": 2, "structure": 3}
TASK_NAMES = {v: k for k, v in TASK_LABEL.items()}


def _parse_task_list(value: str, *, env_name: str) -> tuple[str, ...]:
    aliases = {"segmentation": "seg", "struct": "structure"}
    if not value.strip():
        return tuple(TASK_LABEL)
    tasks = tuple(aliases.get(task.strip(), task.strip())
                  for task in value.split(",") if task.strip())
    unknown = sorted(set(tasks) - TASK_LABEL.keys())
    if unknown:
        raise ValueError(f"{env_name} contains unknown task(s): {', '.join(unknown)}")
    if not tasks:
        raise ValueError(f"{env_name} must name at least one task")
    return tuple(dict.fromkeys(tasks))


ACTIVE_TASKS = _parse_task_list(os.environ.get("VB_TASKS", ""), env_name="VB_TASKS")
RESET_TASKS = _parse_task_list(os.environ.get("VB_RESET_TASKS", ""),
                                env_name="VB_RESET_TASKS") if os.environ.get("VB_RESET_TASKS", "").strip() else ()

OUT.mkdir(parents=True, exist_ok=True)
LOG_F = open(OUT / "train.log", "a", buffering=1)


class EMA:
    """Exponential Moving Average for model parameters."""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self._register()

    def _register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone().detach()

    @torch.no_grad()
    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self):
        """Swap model params with EMA shadow. Call again to swap back."""
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                # Atomic swap: keep both copies
                orig = p.data.clone()
                p.data.copy_(self.shadow[name])
                self.shadow[name] = orig

    def state_dict(self):
        return {"shadow": {k: v.clone() for k, v in self.shadow.items()},
                "decay": self.decay}

    def load_state_dict(self, sd):
        self.decay = sd.get("decay", self.decay)
        device = next(self.model.parameters()).device
        for k, v in sd["shadow"].items():
            if k in self.shadow:
                self.shadow[k] = v.to(device).clone()


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_F.write(line + "\n")


# ── Model ─────────────────────────────────────────────────────────────────
class MARVBMultiTask(nn.Module):
    """MAR-VB with per-task segment tokens and class conditioning."""

    def __init__(self, base: nn.Module, num_tasks: int = 4):
        super().__init__()
        self.m = base
        d = base.encoder_blocks[0].attn.qkv.in_features
        self.num_tasks = num_tasks
        # Per-task learnable context/target segment tokens
        self.ctx_seg = nn.Parameter(torch.zeros(num_tasks, 1, d))
        self.tgt_seg = nn.Parameter(torch.zeros(num_tasks, 1, d))
        nn.init.trunc_normal_(self.ctx_seg, std=0.02)
        nn.init.trunc_normal_(self.tgt_seg, std=0.02)

    def _tiled(self, pos: torch.Tensor) -> torch.Tensor:
        b = self.m.buffer_size
        return torch.cat([pos[:, :b], pos[:, b:], pos[:, b:]], dim=1)

    def _get_segs(self, labels):
        """labels: [B] int → ctx_seg/tgt_seg: [B, 1, D]"""
        return self.ctx_seg[labels], self.tgt_seg[labels]

    def encode_vb(self, ctx_tok, tgt_tok, mask_tgt, class_embedding, labels):
        m = self.m
        ctx_s, tgt_s = self._get_segs(labels)
        x_ctx = m.z_proj(ctx_tok) + ctx_s
        x_tgt = m.z_proj(tgt_tok) + tgt_s
        x = torch.cat([x_ctx, x_tgt], dim=1)
        bsz = x.size(0)
        buf = torch.zeros(bsz, m.buffer_size, x.size(2), device=x.device)
        x = torch.cat([buf, x], dim=1)
        zero_mask = torch.zeros(bsz, m.buffer_size + m.seq_len, device=x.device)
        mask_full = torch.cat([zero_mask, mask_tgt], dim=1)

        if self.training:
            drop = (torch.rand(bsz, device=x.device) < m.label_drop_prob)
            drop = drop.unsqueeze(-1).to(x.dtype)
            class_embedding = drop * m.fake_latent + (1 - drop) * class_embedding
        x[:, :m.buffer_size] = class_embedding.unsqueeze(1)

        x = x + self._tiled(m.encoder_pos_embed_learned)
        x = m.z_proj_ln(x)
        x = x[(1 - mask_full).nonzero(as_tuple=True)].reshape(bsz, -1, x.size(2))
        for blk in m.encoder_blocks:
            x = blk(x)
        return m.encoder_norm(x)

    def decode_vb(self, x, mask_tgt):
        m = self.m
        x = m.decoder_embed(x)
        bsz = x.size(0)
        zero_mask = torch.zeros(bsz, m.buffer_size + m.seq_len, device=x.device)
        mask_full = torch.cat([zero_mask, mask_tgt], dim=1)
        mask_tokens = m.mask_token.repeat(bsz, mask_full.size(1), 1).to(x.dtype)
        padded = mask_tokens.clone()
        padded[(1 - mask_full).nonzero(as_tuple=True)] = x.reshape(-1, x.size(2))
        padded = padded + self._tiled(m.decoder_pos_embed_learned)
        for blk in m.decoder_blocks:
            padded = blk(padded)
        padded = m.decoder_norm(padded)
        z_tgt = padded[:, m.buffer_size + m.seq_len:]
        return z_tgt + m.diffusion_pos_embed_learned

    def forward(self, ctx_lat, tgt_lat, labels):
        m = self.m
        ctx_tok = m.patchify(ctx_lat)
        tgt_tok = m.patchify(tgt_lat)
        gt = tgt_tok.clone().detach()
        class_embedding = m.class_emb(labels)
        orders = m.sample_orders(bsz=ctx_lat.size(0))
        mask = m.random_masking(tgt_tok, orders)
        x = self.encode_vb(ctx_tok, tgt_tok, mask, class_embedding, labels)
        z = self.decode_vb(x, mask)
        return m.forward_loss(z=z, target=gt, mask=mask)

    @torch.no_grad()
    def sample_vb(self, ctx_lat, labels, num_iter=32, cfg=2.0,
                  cfg_schedule="constant", temperature=1.0):
        m = self.m
        bsz = ctx_lat.size(0)
        ctx_tok = m.patchify(ctx_lat)
        tokens = torch.zeros(bsz, m.seq_len, m.token_embed_dim, device=ctx_lat.device)
        mask = torch.ones(bsz, m.seq_len, device=ctx_lat.device)
        orders = m.sample_orders(bsz)
        for step in range(num_iter):
            cur = tokens.clone()
            ce = m.class_emb(labels)
            if cfg != 1.0:
                ce = torch.cat([ce, m.fake_latent.repeat(bsz, 1)], dim=0)
                ctx_tok = torch.cat([ctx_tok, ctx_tok], dim=0)
                tokens = torch.cat([tokens, tokens], dim=0)
                mask = torch.cat([mask, mask], dim=0)
                labels = torch.cat([labels, labels], dim=0)
            x = self.encode_vb(ctx_tok, tokens, mask, ce, labels)
            z = self.decode_vb(x, mask)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(m.seq_len * mask_ratio)],
                                    dtype=torch.float32, device=ctx_lat.device)
            mask_len = torch.maximum(torch.tensor([1.], device=ctx_lat.device),
                                     torch.minimum(mask.sum(-1, keepdim=True) - 1, mask_len))
            mask_next = mask_by_order(mask_len[0], orders, bsz, m.seq_len)
            mask_to_pred = mask[:bsz].bool() if step >= num_iter - 1 else \
                torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
            mask = mask_next
            if cfg != 1.0:
                mask_to_pred = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            z = z[mask_to_pred.nonzero(as_tuple=True)]
            cfg_iter = cfg if cfg_schedule == "constant" else \
                1 + (cfg - 1) * (m.seq_len - mask_len[0]) / m.seq_len
            sampled = m.diffloss.sample(z, temperature, cfg_iter)
            if cfg != 1.0:
                sampled, _ = sampled.chunk(2, dim=0)
                mask_to_pred, _ = mask_to_pred.chunk(2, dim=0)
                labels = labels[:bsz]
            cur[mask_to_pred.nonzero(as_tuple=True)] = sampled
            tokens = cur.clone()
            if cfg != 1.0:
                ctx_tok = ctx_tok[:bsz]
        return m.unpatchify(tokens)


# ── Data ──────────────────────────────────────────────────────────────────
class BaseTaskDataset(Dataset):
    """Base for single-task image→target dataset."""
    def __init__(self, pairs, task_label):
        self.pairs = pairs  # list of (img_path, tgt_path)
        self.label = task_label

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, p: Path, *, resample=Image.BICUBIC) -> torch.Tensor:
        try:
            im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), resample)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, i):
        img_p, tgt_p = self.pairs[i]
        target_resample = Image.NEAREST if self.label == TASK_LABEL["seg"] else Image.BICUBIC
        return self._load_rgb(img_p), self._load_rgb(tgt_p, resample=target_resample), self.label


class StructureTaskDataset(Dataset):
    """Structure task: 16-bit grayscale target → RGB."""
    def __init__(self, pairs, task_label):
        self.pairs = pairs
        self.label = task_label

    def __len__(self):
        return len(self.pairs)

    def _load_rgb(self, p: Path) -> torch.Tensor:
        try:
            im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    def _load_gray(self, p: Path) -> torch.Tensor:
        """16-bit grayscale → float32 tensor [0,1] → replicate to RGB."""
        try:
            im = Image.open(p).resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        except Exception:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE)
        arr = np.asarray(im, dtype=np.float32)
        # Normalize to [0, 1]
        vmin, vmax = arr.min(), arr.max()
        if vmax - vmin > 1e-8:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)
        # Convert to RGB [-1, 1]
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(np.stack([arr] * 3, axis=0))

    def __getitem__(self, i):
        img_p, tgt_p = self.pairs[i]
        return self._load_rgb(img_p), self._load_gray(tgt_p), self.label


class MultiTaskWrapper(Dataset):
    """Wraps multiple task datasets with WeightedRandomSampler-based indexing.

    Each task dataset produces (img, tgt, label). This wrapper concatenates
    them and provides uniform indexing for WeightedRandomSampler.
    """
    def __init__(self, datasets):
        self.datasets = datasets
        self.cum_lens = []
        total = 0
        for ds in datasets:
            total += len(ds)
            self.cum_lens.append(total)

    def __len__(self):
        return self.cum_lens[-1]

    def __getitem__(self, idx):
        # Binary search to find which dataset this index belongs to
        for i, cum in enumerate(self.cum_lens):
            if idx < cum:
                local_idx = idx - (self.cum_lens[i - 1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        return self.datasets[-1][-1]  # fallback


def _build_manifest_data(manifest_path: Path):
    """Load explicit annotation-backed pairs without target-as-input leakage."""
    rows = []
    with manifest_path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("task") not in TASK_LABEL:
                raise ValueError(f"unknown task on manifest line {line_no}: {row.get('task')}")
            if row.get("split") not in {"train", "val"}:
                raise ValueError(f"invalid split on manifest line {line_no}: {row.get('split')}")
            inp, tgt = Path(row["input"]), Path(row["target"])
            if not inp.exists() or not tgt.exists():
                raise FileNotFoundError(f"missing manifest pair on line {line_no}: {inp} -> {tgt}")
            rows.append(row)

    max_train = int(os.environ.get("VB_MANIFEST_MAX_TRAIN_PER_TASK", "0"))
    # Keep frequent in-training validation practical on MPS. The standalone
    # evaluator consumes the complete held-out manifest after a run.
    max_val = int(os.environ.get("VB_MANIFEST_MAX_VAL_PER_TASK", "32"))
    train_ds_list, val_ds_list = [], []
    for task in ACTIVE_TASKS:
        train_rows = [r for r in rows if r["task"] == task and r["split"] == "train"]
        val_rows = [r for r in rows if r["task"] == task and r["split"] == "val"]
        if max_train > 0:
            train_rows = train_rows[:max_train]
        if max_val > 0:
            val_rows = val_rows[:max_val]
        train_pairs = [(Path(r["input"]), Path(r["target"])) for r in train_rows]
        val_pairs = [(Path(r["input"]), Path(r["target"])) for r in val_rows]
        if train_pairs and val_pairs:
            train_ds_list.append(BaseTaskDataset(train_pairs, TASK_LABEL[task]))
            val_ds_list.append(BaseTaskDataset(val_pairs, TASK_LABEL[task]))
        log(f"  manifest {task}: train={len(train_pairs)} val={len(val_pairs)}")

    if not train_ds_list:
        raise RuntimeError(f"No usable pairs in manifest: {manifest_path}")
    train_ds = MultiTaskWrapper(train_ds_list)
    # No replacement: each clean pair is seen at most once per DataLoader epoch.
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    val_ds = MultiTaskWrapper(val_ds_list)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    log(f"  manifest={manifest_path} train={len(train_ds)} val={len(val_ds)} "
        f"batches={len(train_dl)}")
    return train_dl, val_dl


def build_data():
    if PAIR_MANIFEST:
        return _build_manifest_data(Path(PAIR_MANIFEST))
    img_root = Path(DATA_ROOT) / "images"
    tgt_root = Path(DATA_ROOT) / "targets"
    rng = np.random.RandomState(SEED)
    train_ds_list, val_ds_list = [], []

    def _filter_pairs(img_dir, tgt_dir, task, n_train, n_val):
        imgs = {p.stem: p for ext in ("*.jpg", "*.png") for p in img_dir.glob(ext)}
        tgt_files = sorted(tgt_dir.glob("*.png"))
        tgts = {p.stem for p in tgt_files}
        stems = sorted(set(imgs.keys()) & tgts)
        n_paired = len(stems)

        # For cross-domain tasks (seg, structure): random image↔target pairing
        if n_paired == 0 and len(tgt_files) > 0:
            img_list = sorted(imgs.values())
            log(f"  {task}: images={len(imgs)} targets={len(tgts)} "
                f"paired=0 (cross-domain, random pairing)")
            n_tr = min(n_train, max(len(tgts), len(imgs)))
            n_vl = min(max(n_val, 40), n_tr // 10)
            # Create random pairs
            idx_img = rng.choice(len(img_list), n_train + n_val, replace=True)
            idx_tgt = rng.choice(len(tgt_files), n_train + n_val, replace=True)
            train_pairs = [(img_list[idx_img[i]], tgt_files[idx_tgt[i]])
                          for i in range(n_tr)]
            val_pairs = [(img_list[idx_img[n_tr + i]], tgt_files[idx_tgt[n_tr + i]])
                        for i in range(n_vl)]
            return train_pairs, val_pairs

        log(f"  {task}: images={len(imgs)} targets={len(tgts)} paired={n_paired}")
        rng.shuffle(stems)
        val_s = stems[:n_val]
        train_s = stems[n_val:n_val + n_train]
        train_pairs = [(imgs[s], tgt_dir / f"{s}.png") for s in train_s]
        val_pairs = [(imgs[s], tgt_dir / f"{s}.png") for s in val_s]
        return train_pairs, val_pairs

    task_configs = [
        ("depth", img_root / "depth", tgt_root / "depth"),
        ("normal", img_root / "depth", tgt_root / "normal"),  # normal uses same COCO input as depth
        ("seg", img_root / "segmentation", tgt_root / "segmentation"),
        # Prefer self-supervised aligned structure targets (Canny+LSD) when available
        ("structure", img_root / "structure",
         tgt_root / "structure_aligned" if (tgt_root / "structure_aligned").exists()
         else tgt_root / "structure"),
    ]

    # Update structure dataset class based on whether targets are aligned RGB
    # (structure_aligned is RGB 3-channel; original structure was 16-bit sparse depth)
    structure_uses_aligned = (tgt_root / "structure_aligned").exists()
    if structure_uses_aligned:
        log(f"  structure: using *aligned* targets (Canny+LSD, 4662 stem-matched pairs)")

    for task, img_dir, tgt_dir in task_configs:
        if not tgt_dir.exists():
            log(f"  {task}: targets/ dir not found, skipping")
            continue
        n_tr = int(N_TRAIN * TASK_WEIGHTS[task] / sum(TASK_WEIGHTS.values()))
        # Use N_VAL_PER_TASK for all tasks to ensure balanced val coverage
        n_vl = N_VAL_PER_TASK
        train_pairs, val_pairs = _filter_pairs(img_dir, tgt_dir, task, n_tr, n_vl)

        ds_cls = BaseTaskDataset  # all tasks now use BaseTaskDataset (RGB 3-channel)
        if train_pairs:
            train_ds_list.append(ds_cls(train_pairs, TASK_LABEL[task]))
            val_ds_list.append(ds_cls(val_pairs, TASK_LABEL[task]))

    if not train_ds_list:
        raise RuntimeError("No task data available!")

    train_ds = MultiTaskWrapper(train_ds_list)
    # Weighted sampler: each sample equally likely regardless of task
    sample_weights = torch.ones(len(train_ds))
    train_sampler = WeightedRandomSampler(sample_weights, len(train_ds),
                                          replacement=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, sampler=train_sampler,
                          num_workers=NUM_WORKERS, drop_last=True)

    val_ds = MultiTaskWrapper(val_ds_list)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    log(f"  train={len(train_ds)} val={len(val_ds)} "
        f"batches={len(train_dl)}")
    return train_dl, val_dl


# ── Train ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_latent(vae, imgs, sample=True):
    posterior = vae.encode(imgs)
    latent = posterior.sample() if sample else posterior.mode()
    return latent.mul_(VAE_SCALE)


def _interp_pos_grid(pe, new_hw):
    L, D = pe.shape[1], pe.shape[2]
    old_hw = int(round(L ** 0.5))
    g = pe.reshape(1, old_hw, old_hw, D).permute(0, 3, 1, 2)
    g = torch.nn.functional.interpolate(g, size=(new_hw, new_hw),
                                        mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, D)


def resize_pos_keys(sd, prefix, hw):
    buf = 64
    for k in (prefix + "encoder_pos_embed_learned",
              prefix + "decoder_pos_embed_learned"):
        sd[k] = torch.cat([sd[k][:, :buf],
                           _interp_pos_grid(sd[k][:, buf:], hw)], dim=1)
    sd[prefix + "diffusion_pos_embed_learned"] = \
        _interp_pos_grid(sd[prefix + "diffusion_pos_embed_learned"], hw)
    return sd


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"device={DEVICE} steps={STEPS} batch={BATCH} lr={LR} out={OUT}")
    log(f"data_root={DATA_ROOT} img_size={IMG_SIZE} total_steps={TOTAL_STEPS}")
    log(f"lr={LR} wd={WEIGHT_DECAY} dropout={DROPOUT} ema={EMA_DECAY}")
    log(f"task_weights={TASK_WEIGHTS}")
    log(f"active_tasks={ACTIVE_TASKS}")

    base = mar_large(buffer_size=64, diffloss_d=8, diffloss_w=1280,
                    num_sampling_steps="100", img_size=IMG_SIZE,
                    attn_dropout=DROPOUT, proj_dropout=DROPOUT)
    sd0 = load_file(f"{CKPT_DIR}/mar-large.safetensors")
    if IMG_SIZE != 256:
        sd0 = resize_pos_keys(sd0, "", IMG_SIZE // 16)
    base.load_state_dict(sd0, strict=True)
    model = MARVBMultiTask(base, num_tasks=len(TASK_LABEL)).to(DEVICE).train()
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    log(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M "
        f"trainable ({len(TASK_LABEL)} tasks)")

    ema = EMA(model, decay=EMA_DECAY)
    log(f"ema_decay={EMA_DECAY}")

    train_dl, val_dl = build_data()
    log(f"train_dl: {len(train_dl)} batches/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    def lr_at(step):
        eff_step = step - start_step  # relative to resume point
        if eff_step <= 0:
            return LR * 0.01
        if eff_step < WARMUP:
            return LR * eff_step / max(1, WARMUP)
        t = (eff_step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
        return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    @torch.no_grad()
    def val_loss(eval_step=0):
        # Keep validation comparable: fixed masking/order RNG and posterior mode.
        torch.manual_seed(SEED + 100000 + int(eval_step))
        np.random.seed(SEED + int(eval_step))
        ema.apply()
        model.eval()
        losses, task_losses = [], {TASK_LABEL[t]: [] for t in TASK_LABEL}
        for ds in val_dl.dataset.datasets:
            dl = DataLoader(ds, batch_size=BATCH, shuffle=False)
            for im, tg, lb in dl:
                im, tg, lb = im.to(DEVICE), tg.to(DEVICE), lb.to(DEVICE)
                l = model(encode_latent(vae, im, sample=False),
                          encode_latent(vae, tg, sample=False), lb).item()
                losses.append(l)
                for t in lb.unique():
                    mask = lb == t
                    if mask.sum() > 0:
                        task_losses[int(t.item())].append(l)
        model.train()
        ema.apply()  # restore original weights
        avg = float(np.mean(losses))
        task_avg = {TASK_NAMES.get(k, str(k)): float(np.mean(v)) if v else 0.0
                    for k, v in task_losses.items()}
        return avg, task_avg

    @torch.no_grad()
    def sample_panel(step):
        if DEVICE == "mps":
            torch.mps.empty_cache()
        torch.manual_seed(SEED + 200000 + int(step))
        np.random.seed(SEED + int(step))
        ema.apply()
        model.eval()
        panels = []
        task_maes = {}
        # Sample from each task's val split
        for ds in val_dl.dataset.datasets:
            if len(ds) == 0:
                continue
            idx = np.random.choice(min(len(ds), 2))
            im, tg, lb = ds[idx]
            im = im.unsqueeze(0).to(DEVICE)
            tg = tg.unsqueeze(0).to(DEVICE)
            lb = torch.tensor([ds.label], device=DEVICE)
            ctx = encode_latent(vae, im, sample=False)
            lat = model.sample_vb(ctx, lb, num_iter=SAMPLE_ITER, cfg=2.0)
            pred = vae.decode(lat / VAE_SCALE).clamp(-1, 1)
            task_maes[TASK_NAMES[ds.label]] = float(
                ((pred + 1) / 2 - (tg + 1) / 2).abs().mean().item())
            panels.append(torch.cat([im, tg, pred], dim=0))
        if panels:
            panel = torch.cat(panels, dim=0)
            save_image(panel, OUT / f"panel_step{step}.png", nrow=3,
                       normalize=True, value_range=(-1, 1))
            mae = float(np.mean(list(task_maes.values())))
            task_str = " ".join(f"{t}={v:.4f}" for t, v in task_maes.items())
            log(f"[sample] step {step} global_mae={mae:.4f} [{task_str}]")
        model.train()
        ema.apply()  # restore original

    best_val = float("inf")
    start_step = 0
    if RESUME and Path(RESUME).exists():
        sd = torch.load(RESUME, map_location="cpu", weights_only=False)
        msd = sd["model"]
        # Resize pos embeds if shape doesn't match (cross img_size fine-tuning)
        cur_diff_shape = model.m.diffusion_pos_embed_learned.shape
        if msd["m.diffusion_pos_embed_learned"].shape != cur_diff_shape:
            cur_hw = model.m.diff_pos_embed_h if hasattr(model.m, "diff_pos_embed_h") else None
            # Compute current hw from pos embed shape
            cur_total = model.m.encoder_pos_embed_learned.shape[1] - model.m.buffer_size
            cur_hw_val = int(round(cur_total ** 0.5))
            log(f"[resize] pos_embeds {msd['m.diffusion_pos_embed_learned'].shape[1]} -> "
                f"{cur_total} (hw={cur_hw_val})")
            msd = resize_pos_keys(msd, "m.", cur_hw_val)
        # Handle multi-task vs single-task checkpoint compatibility
        try:
            model.load_state_dict(msd, strict=True)
        except RuntimeError:
            # Allow missing multi-task specific params (init from scratch)
            model.load_state_dict(msd, strict=False)
            log("[resume] loaded with strict=False (multi-task expansion)")
        if RESET_TASKS:
            with torch.no_grad():
                for task in RESET_TASKS:
                    label = TASK_LABEL[task]
                    nn.init.trunc_normal_(model.ctx_seg.data[label], std=0.02)
                    nn.init.trunc_normal_(model.tgt_seg.data[label], std=0.02)
            log(f"[resume] task segment tokens RESET: {', '.join(RESET_TASKS)}")
        if "opt" in sd and not RESET_OPT:
            # Check if opt state shapes match current model (no pos embed resize)
            try:
                opt_state = sd["opt"]
                state_keys = list(opt_state["state"].keys())
                # Get current param shapes (ordered)
                model_param_shapes = {id(p): tuple(p.shape) for p in model.parameters()}
                # Check all exp_avg shapes match
                mismatch = False
                for k in state_keys[:5]:  # sample first 5
                    s = opt_state["state"][k]
                    if "exp_avg" in s:
                        if tuple(s["exp_avg"].shape) not in model_param_shapes.values():
                            mismatch = True
                            break
                if mismatch:
                    log(f"[resume] opt state shape mismatch (pos embed resize), "
                        f"skipping opt state (fresh Adam moments)")
                else:
                    opt.load_state_dict(opt_state)
            except Exception as e:
                log(f"[resume] optimizer state skipped: {str(e)[:80]}")
        elif RESET_OPT:
            log("[resume] optimizer state RESET")
        start_step = sd["step"]
        best_val = sd.get("val_loss", best_val)
        if RESET_BEST:
            best_val = float("inf")
            log("[resume] best validation RESET")
        if "ema" in sd and not RESET_EMA:
            ema.load_state_dict(sd["ema"])
            if RESET_TASKS:
                for task in RESET_TASKS:
                    label = TASK_LABEL[task]
                    ema.shadow["ctx_seg"][label].copy_(model.ctx_seg.data[label])
                    ema.shadow["tgt_seg"][label].copy_(model.tgt_seg.data[label])
            log(f"[resume] EMA shadow loaded from ckpt")
        else:
            if RESET_EMA:
                # Re-register after the checkpoint model weights are loaded;
                # keeping the pre-resume shadow would swap in base weights at
                # the first validation/sample call.
                ema = EMA(model, decay=EMA_DECAY)
                log(f"[resume] EMA shadow RESET (training new task distribution)")
        log(f"[resume] {RESUME} step={start_step} val_loss={best_val:.4f}")

    train_iter = iter(train_dl)
    t0 = time.time()
    loss_hist: list[float] = []

    for step in range(start_step + 1, STEPS + 1):
        try:
            im, tg, lb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            im, tg, lb = next(train_iter)
        im, tg, lb = im.to(DEVICE), tg.to(DEVICE), lb.to(DEVICE)

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = model(encode_latent(vae, im), encode_latent(vae, tg), lb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        ema.update()
        loss_hist.append(loss.item())

        # MPS memory management: clear every step to prevent fragmentation OOM
        if DEVICE == "mps" and step % 5 == 0:
            torch.mps.empty_cache()

        if step % 20 == 0:
            lb_counts = {int(t.item()): (lb == t).sum().item()
                         for t in lb.unique()}
            lb_str = " ".join(f"{TASK_NAMES.get(k, str(k))}={v}"
                              for k, v in sorted(lb_counts.items()))
            log(f" step {step}/{STEPS} loss={np.mean(loss_hist[-20:]):.4f} "
                f"lr={lr_at(step):.2e} gn={gn:.2f} [{lb_str}] "
                f"({(time.time()-t0)/max(1, step-start_step):.2f}s/it)")
        if step % VAL_EVERY == 0:
            vl, task_vl = val_loss(step)
            task_str = " ".join(f"{t}={v:.4f}" for t, v in task_vl.items())
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "ema": ema.state_dict(),
                            "step": step, "val_loss": vl},
                           OUT / "ckpt_best.pt")
                log(f"[val] step {step} val_loss={vl:.4f} *BEST* [{task_str}]")
            else:
                log(f"[val] step {step} val_loss={vl:.4f} best={best_val:.4f} [{task_str}]")
            # Save ckpt_latest AFTER val to keep best_val fresh
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step, "val_loss": best_val},
                       OUT / "ckpt_latest.pt")
        elif SAVE_LATEST_EVERY > 0 and step % SAVE_LATEST_EVERY == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(),
                        "step": step, "val_loss": best_val},
                       OUT / "ckpt_latest.pt")
        if not SAMPLE_SKIP and (step % SAMPLE_EVERY == 0 or step == STEPS):
            sample_panel(step)

    if not SKIP_FINAL_SAVE:
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.state_dict(),
                    "step": STEPS, "val_loss": best_val}, OUT / "ckpt_final.pt")
    else:
        log("[done] final checkpoint skipped (VB_SKIP_FINAL_SAVE=1)")
    json.dump({"best_val_loss": best_val, "steps": STEPS, "batch": BATCH,
               "num_workers": NUM_WORKERS,
               "save_latest_every": SAVE_LATEST_EVERY, "lr": LR,
               "data": "vb_snvc_train_v5_full", "task": "multitask",
               "tasks": list(TASK_LABEL.keys()),
               "active_tasks": list(ACTIVE_TASKS),
               "task_weights": TASK_WEIGHTS,
               "pair_manifest": PAIR_MANIFEST or None,
               "resume": RESUME or None,
               "reset_opt": RESET_OPT,
               "reset_ema": RESET_EMA,
               "reset_best": RESET_BEST,
               "reset_tasks": list(RESET_TASKS),
               "skip_final_save": SKIP_FINAL_SAVE,
               "manifest_max_val_per_task": int(
                   os.environ.get("VB_MANIFEST_MAX_VAL_PER_TASK", "32"))},
              open(OUT / "train_config.json", "w"), indent=2)
    log(f"[done] {(time.time()-t0)/60:.1f} min, best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
