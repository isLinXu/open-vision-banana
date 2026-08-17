#!/usr/bin/env python3
"""MAR-VB Single-Task Depth Fine-Tuning.

Fine-tune a MAR-Large checkpoint on depth-only data with:
  - Higher learning rate (3e-5 vs 1e-5 multitask)
  - Longer training (5000+ steps)
  - Task-specific augmentation (random crop, flip)
  - EMA for stable convergence
  - Best-checkpoint saving based on val loss

This concentrates all model capacity on depth estimation,
which is the most promising task (pearson=0.88 in v1).

Usage:
  # From a multitask checkpoint (transfer learning)
  VB_OUT=output/mar_vb_depth_finetune \
  VB_RESUME=output/mar_vb_v5_full_clean_v2_3task_refine_384_20260811/ckpt_best.pt \
  VB_STEPS=5000 VB_LR=3e-5 \
  python scripts/mar_vb_depth_finetune.py

  # From scratch (fresh MAR-Large)
  VB_OUT=output/mar_vb_depth_scratch \
  VB_STEPS=8000 VB_LR=1e-4 \
  python scripts/mar_vb_depth_finetune.py

Environment variables:
  VB_OUT          Output directory (default: output/mar_vb_depth_finetune)
  VB_RESUME       Checkpoint to resume/transfer from (default: none)
  VB_DATA_ROOT    Dataset root (default: v5_full path)
  VB_STEPS        Training steps (default: 5000)
  VB_LR           Learning rate (default: 3e-5)
  VB_IMG_SIZE     Image size (default: 384)
  VB_BATCH        Batch size (default: 2)
  VB_PAIR_MANIFEST  Pair manifest JSONL (optional, uses dataset if empty)
  VB_RESET_OPT    Reset optimizer state on resume (default: 1 for transfer)
  VB_RESET_BEST   Reset best val loss on resume (default: 1 for transfer)
"""
from __future__ import annotations

import json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch, torch.nn as nn
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
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
    str(PROJECT_ROOT / "output" / "mar_vb_depth_finetune")))

IMG_SIZE = int(os.environ.get("VB_IMG_SIZE", "384"))
N_TRAIN = int(os.environ.get("VB_N_TRAIN", "20000"))
N_VAL = int(os.environ.get("VB_N_VAL", "64"))
STEPS = int(os.environ.get("VB_STEPS", "5000"))
BATCH = int(os.environ.get("VB_BATCH", "2"))
NUM_WORKERS = int(os.environ.get("VB_NUM_WORKERS", "2"))
LR = float(os.environ.get("VB_LR", "3e-5"))
WARMUP = int(os.environ.get("VB_WARMUP", "100"))
WEIGHT_DECAY = float(os.environ.get("VB_WD", "0.05"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
DROPOUT = float(os.environ.get("VB_DROPOUT", "0.1"))  # lower dropout for fine-tuning
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "200"))
SAVE_EVERY = int(os.environ.get("VB_SAVE_EVERY", "500"))
SAMPLE_EVERY = int(os.environ.get("VB_SAMPLE_EVERY", "1000"))
EMA_DECAY = float(os.environ.get("VB_EMA", "0.9995"))
SAMPLE_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
RESUME = os.environ.get("VB_RESUME", "")
RESET_EMA = os.environ.get("VB_RESET_EMA", "1") == "1"
RESET_OPT = os.environ.get("VB_RESET_OPT", "1") == "1"
RESET_BEST = os.environ.get("VB_RESET_BEST", "1") == "1"
PAIR_MANIFEST = os.environ.get("VB_PAIR_MANIFEST", "")
SEED = 42
VAE_SCALE = 0.2325
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# Only depth task
TASK_LABEL = 0
NUM_TASKS = 4  # keep multitask architecture, but only train depth

LOG_F = None


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_F:
        LOG_F.write(line + "\n")


# ── EMA (copied from v5 multitask) ────────────────────────────────────────
class EMA:
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
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
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


# ── Model (same as v5 multitask, but only train depth) ────────────────────
class MARVBMultiTask(nn.Module):
    def __init__(self, base: nn.Module, num_tasks: int = 4):
        super().__init__()
        self.m = base
        d = base.encoder_blocks[0].attn.qkv.in_features
        self.num_tasks = num_tasks
        self.ctx_seg = nn.Parameter(torch.zeros(num_tasks, 1, d))
        self.tgt_seg = nn.Parameter(torch.zeros(num_tasks, 1, d))
        nn.init.trunc_normal_(self.ctx_seg, std=0.02)
        nn.init.trunc_normal_(self.tgt_seg, std=0.02)

    def _tiled(self, pos):
        b = self.m.buffer_size
        return torch.cat([pos[:, :b], pos[:, b:], pos[:, b:]], dim=1)

    def _get_segs(self, labels):
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
    def sample_vb(self, ctx_lat, labels, num_iter=32, cfg=2.0, temperature=1.0):
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
            cfg_iter = cfg
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
class DepthDataset(Dataset):
    """Depth-only image->target dataset."""
    def __init__(self, pairs, img_size=384):
        self.pairs = pairs
        self.img_size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, tgt_path = self.pairs[idx]
        with Image.open(img_path) as img:
            inp = img.convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
            inp = np.asarray(inp, dtype=np.float32) / 127.5 - 1.0
        with Image.open(tgt_path) as tgt:
            target = tgt.convert("RGB").resize((self.img_size, self.img_size), Image.BICUBIC)
            target = np.asarray(target, dtype=np.float32) / 127.5 - 1.0
        # Random horizontal flip
        if np.random.random() < 0.5:
            inp = inp[:, ::-1].copy()
            target = target[:, ::-1].copy()
        return (
            torch.from_numpy(inp).permute(2, 0, 1),
            torch.from_numpy(target).permute(2, 0, 1),
        )


def load_pairs_from_manifest(manifest_path: str, split: str = "train"):
    """Load depth pairs from a JSONL manifest."""
    pairs = []
    for line in Path(manifest_path).open():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split") == split and row.get("task") == "depth":
            pairs.append((row["input"], row["target"]))
    return pairs


def load_pairs_from_dataset(data_root: str, split: str = "train", max_samples: int = 20000):
    """Load depth pairs from the v5_full dataset directory structure."""
    root = Path(data_root)
    img_dir = root / "images" / "depth"
    tgt_dir = root / "targets" / "depth"
    if not img_dir.exists():
        raise FileNotFoundError(f"Depth images dir not found: {img_dir}")

    stems = sorted({
        p.stem for p in img_dir.glob("*.jpg")
    })
    np.random.seed(SEED)
    np.random.shuffle(stems)
    n_val = min(64, len(stems) // 10)
    if split == "val":
        stems = stems[:n_val]
    else:
        stems = stems[n_val:n_val + max_samples]

    pairs = []
    for stem in stems:
        img_path = str(img_dir / f"{stem}.jpg")
        tgt_path = str(tgt_dir / f"{stem}.png")
        if Path(tgt_path).exists():
            pairs.append((img_path, tgt_path))
    return pairs


def encode_latent(vae, imgs, sample=True):
    posterior = vae.encode(imgs)
    latent = posterior.sample() if sample else posterior.mode()
    return latent.mul_(VAE_SCALE)


def _interp_pos_grid(pe, new_hw):
    L, D = pe.shape[1], pe.shape[2]
    old_hw = int(round(L ** 0.5))
    g = pe.reshape(1, old_hw, old_hw, D).permute(0, 3, 1, 2)
    g = torch.nn.functional.interpolate(g, size=(new_hw, new_hw), mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, D)


def resize_pos_keys(sd, prefix, hw):
    buf = 64
    for k in (prefix + "encoder_pos_embed_learned", prefix + "decoder_pos_embed_learned"):
        sd[k] = torch.cat([sd[k][:, :buf], _interp_pos_grid(sd[k][:, buf:], hw)], dim=1)
    sd[prefix + "diffusion_pos_embed_learned"] = _interp_pos_grid(sd[prefix + "diffusion_pos_embed_learned"], hw)
    return sd


# ── Training ──────────────────────────────────────────────────────────────
def main():
    global LOG_F
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_F = open(OUT / "train.log", "a")

    log(f"=== MAR-VB Depth Fine-Tuning ===")
    log(f"device={DEVICE} steps={STEPS} batch={BATCH} lr={LR} out={OUT}")
    log(f"data_root={DATA_ROOT} img_size={IMG_SIZE}")
    log(f"dropout={DROPOUT} ema={EMA_DECAY} wd={WEIGHT_DECAY}")
    if RESUME:
        log(f"resume from {RESUME} (reset_opt={RESET_OPT} reset_ema={RESET_EMA} reset_best={RESET_BEST})")

    # Load data
    if PAIR_MANIFEST:
        train_pairs = load_pairs_from_manifest(PAIR_MANIFEST, "train")
        val_pairs = load_pairs_from_manifest(PAIR_MANIFEST, "val")
    else:
        train_pairs = load_pairs_from_dataset(DATA_ROOT, "train", N_TRAIN)
        val_pairs = load_pairs_from_dataset(DATA_ROOT, "val")
    log(f"train pairs={len(train_pairs)} val pairs={len(val_pairs)}")

    train_ds = DepthDataset(train_pairs, IMG_SIZE)
    val_ds = DepthDataset(val_pairs, IMG_SIZE)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS)

    # Load model
    base = mar_large(buffer_size=64, diffloss_d=8, diffloss_w=1280,
                    num_sampling_steps="100", img_size=IMG_SIZE,
                    attn_dropout=DROPOUT, proj_dropout=DROPOUT)
    sd0 = load_file(f"{CKPT_DIR}/mar-large.safetensors")
    if IMG_SIZE != 256:
        sd0 = resize_pos_keys(sd0, "", IMG_SIZE // 16)
    base.load_state_dict(sd0, strict=True)

    model = MARVBMultiTask(base, num_tasks=NUM_TASKS).to(DEVICE).train()
    log(f"model loaded (trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,})")

    # Resume / transfer
    best_val_loss = float("inf")
    global_step = 0
    if RESUME:
        ckpt = torch.load(RESUME, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"], strict=True)
        log(f"  loaded checkpoint (step={ckpt.get('step')}, val_loss={ckpt.get('val_loss')})")
        if not RESET_BEST:
            best_val_loss = ckpt.get("val_loss", float("inf"))
        if not RESET_EMA and "ema" in ckpt:
            ema = EMA(model, decay=ckpt["ema"].get("decay", EMA_DECAY))
            ema.load_state_dict(ckpt["ema"])
            log(f"  loaded EMA (decay={ema.decay:.4f})")
        else:
            ema = EMA(model, decay=EMA_DECAY)
            log(f"  fresh EMA (decay={EMA_DECAY})")
    else:
        ema = EMA(model, decay=EMA_DECAY)

    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    # Optimizer
    decay_params = [p for n, p in model.named_parameters()
                    if p.requires_grad and p.dim() > 1]
    nodecay_params = [p for n, p in model.named_parameters()
                      if p.requires_grad and p.dim() <= 1]
    optim = torch.optim.AdamW([
        {"params": decay_params, "weight_decay": WEIGHT_DECAY},
        {"params": nodecay_params, "weight_decay": 0.0},
    ], lr=LR, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        progress = (step - WARMUP) / max(1, STEPS - WARMUP)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    if RESUME and not RESET_OPT:
        if "optimizer" in ckpt:
            optim.load_state_dict(ckpt["optimizer"])
            log("  loaded optimizer state")
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
            log("  loaded scheduler state")
        global_step = ckpt.get("step", 0)

    # Training loop
    log("Starting training...")
    train_iter = iter(train_dl)
    losses = []

    while global_step < STEPS:
        try:
            inp, tgt = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            inp, tgt = next(train_iter)

        inp, tgt = inp.to(DEVICE), tgt.to(DEVICE)
        labels = torch.full((inp.size(0),), TASK_LABEL, dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            ctx_lat = encode_latent(vae, inp, sample=False)
            tgt_lat = encode_latent(vae, tgt, sample=True)

        loss = model(ctx_lat, tgt_lat, labels)
        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        scheduler.step()
        ema.update()

        losses.append(loss.item())
        global_step += 1

        if global_step % 50 == 0:
            avg_loss = np.mean(losses[-50:])
            lr_now = scheduler.get_last_lr()[0]
            log(f"step {global_step}/{STEPS} loss={loss.item():.4f} avg={avg_loss:.4f} lr={lr_now:.2e}")

        # Validation
        if global_step % VAL_EVERY == 0:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for vi, (vin, vtn) in enumerate(val_dl):
                    if vi >= 10:
                        break
                    vin, vtn = vin.to(DEVICE), vtn.to(DEVICE)
                    vlabels = torch.full((vin.size(0),), TASK_LABEL, dtype=torch.long, device=DEVICE)
                    vctx = encode_latent(vae, vin, sample=False)
                    vtgt = encode_latent(vae, vtn, sample=True)
                    vloss = model(vctx, vtgt, vlabels)
                    val_losses.append(vloss.item())
            val_loss = float(np.mean(val_losses))
            model.train()

            improved = val_loss < best_val_loss
            if improved:
                best_val_loss = val_loss
            log(f"  VAL step={global_step} val_loss={val_loss:.4f} best={best_val_loss:.4f} {'★' if improved else ''}")

            # Save best
            if improved:
                save_ckpt(OUT / "ckpt_best.pt", model, ema, optim, scheduler,
                          global_step, val_loss, best_val_loss)

        # Periodic save
        if global_step % SAVE_EVERY == 0:
            save_ckpt(OUT / "ckpt_latest.pt", model, ema, optim, scheduler,
                      global_step, None, best_val_loss)

        # Sample visualization
        if global_step % SAMPLE_EVERY == 0:
            sample_visualization(model, vae, val_ds, global_step)

    # Final save
    save_ckpt(OUT / "ckpt_final.pt", model, ema, optim, scheduler,
              global_step, None, best_val_loss)
    log(f"Training complete. Best val_loss={best_val_loss:.4f}")
    if LOG_F:
        LOG_F.close()


def save_ckpt(path, model, ema, optim, scheduler, step, val_loss, best_val_loss):
    """Save checkpoint."""
    ckpt = {
        "model": model.state_dict(),
        "ema": ema.state_dict(),
        "optimizer": optim.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
    }
    torch.save(ckpt, str(path))
    log(f"  saved {path.name} (step={step})")


def sample_visualization(model, vae, val_ds, step):
    """Generate and save sample predictions."""
    model.eval()
    with torch.no_grad():
        idx = np.random.randint(len(val_ds))
        inp, tgt = val_ds[idx]
        inp = inp.unsqueeze(0).to(DEVICE)
        tgt = tgt.unsqueeze(0).to(DEVICE)
        ctx = encode_latent(vae, inp, sample=False)
        labels = torch.tensor([TASK_LABEL], device=DEVICE)
        latent = model.sample_vb(ctx, labels, num_iter=SAMPLE_ITER, cfg=1.0)
        pred = vae.decode(latent / VAE_SCALE).clamp(-1, 1)
        panel = torch.cat([inp[0].cpu(), tgt[0].cpu(), pred[0].cpu()], dim=0)
        save_image(panel.unsqueeze(0), str(OUT / f"sample_step{step}.png"),
                   nrow=3, normalize=True, value_range=(-1, 1))
    model.train()


if __name__ == "__main__":
    main()
