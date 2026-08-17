#!/usr/bin/env python3
"""MAR-VB v5_full: Depth instruction tuning on MAR (mar-base, 208M).

Adapted from mar_vb_train.py for vb_snvc_train_v5_full dataset:
- images/depth/*.{jpg,png} → targets/depth/{stem}.png
- ~12k train samples (vs ~2.2k in v3)

Usage (MPS):
  VB_STEPS=3000 VB_DATA_ROOT=... /Users/gatilin/.workbuddy/binaries/python/envs/default/bin/python scripts/mar_vb_train_v5.py
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
from mar.mar import mar_base, mask_by_order
from mar.vae import AutoencoderKL

# ── Config ────────────────────────────────────────────────────────────────
CKPT_DIR = "/Users/gatilin/MyWork/models/ckpts/mar"
DATA_ROOT = os.environ.get("VB_DATA_ROOT",
    "/Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v5_full")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("VB_OUT", str(PROJECT_ROOT / "output" / "mar_vb_depth_v5_full")))

IMG_SIZE = int(os.environ.get("VB_IMG_SIZE", "256"))
N_TRAIN = int(os.environ.get("VB_N_TRAIN", "11000"))
N_VAL = int(os.environ.get("VB_VAL", "500"))
STEPS = int(os.environ.get("VB_STEPS", "3000"))
TOTAL_STEPS = int(os.environ.get("VB_TOTAL_STEPS", str(STEPS)))
BATCH = int(os.environ.get("VB_BATCH", "8"))
LR = float(os.environ.get("VB_LR", "1e-4"))
WARMUP = int(os.environ.get("VB_WARMUP", "100"))
WEIGHT_DECAY = float(os.environ.get("VB_WD", "0.01"))
GRAD_CLIP = float(os.environ.get("VB_GRAD_CLIP", "1.0"))
VAL_EVERY = int(os.environ.get("VB_VAL_EVERY", "100"))
SAMPLE_EVERY = int(os.environ.get("VB_SAMPLE_EVERY", "500"))
SAMPLE_ITER = int(os.environ.get("VB_SAMPLE_ITER", "32"))
RESUME = os.environ.get("VB_RESUME", "")
SEED = 42
VAE_SCALE = 0.2325
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

OUT.mkdir(parents=True, exist_ok=True)
LOG_F = open(OUT / "train.log", "a", buffering=1)


def log(msg: str):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_F.write(line + "\n")


# ── Model ─────────────────────────────────────────────────────────────────
class MARVB(nn.Module):
    def __init__(self, base: nn.Module):
        super().__init__()
        self.m = base
        d = base.encoder_blocks[0].attn.qkv.in_features
        self.ctx_seg = nn.Parameter(torch.zeros(1, 1, d))
        self.tgt_seg = nn.Parameter(torch.zeros(1, 1, d))

    def _tiled(self, pos: torch.Tensor) -> torch.Tensor:
        b = self.m.buffer_size
        return torch.cat([pos[:, :b], pos[:, b:], pos[:, b:]], dim=1)

    def encode_vb(self, ctx_tok, tgt_tok, mask_tgt, class_embedding):
        m = self.m
        x_ctx = m.z_proj(ctx_tok) + self.ctx_seg
        x_tgt = m.z_proj(tgt_tok) + self.tgt_seg
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
        x = self.encode_vb(ctx_tok, tgt_tok, mask, class_embedding)
        z = self.decode_vb(x, mask)
        return m.forward_loss(z=z, target=gt, mask=mask)

    @torch.no_grad()
    def sample_vb(self, ctx_lat, labels, num_iter=32, cfg=2.0,
                  cfg_schedule="constant", temperature=1.0, progress=False):
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
            x = self.encode_vb(ctx_tok, tokens, mask, ce)
            z = self.decode_vb(x, mask)

            mask_ratio = np.cos(math.pi / 2. * (step + 1) / num_iter)
            mask_len = torch.tensor([np.floor(m.seq_len * mask_ratio)], dtype=torch.float32, device=ctx_lat.device)
            mask_len = torch.maximum(torch.tensor([1.], device=ctx_lat.device),
                                     torch.minimum(mask.sum(-1, keepdim=True) - 1, mask_len))
            mask_next = mask_by_order(mask_len[0], orders, bsz, m.seq_len)
            mask_to_pred = mask[:bsz].bool() if step >= num_iter - 1 else torch.logical_xor(mask[:bsz].bool(), mask_next.bool())
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
            cur[mask_to_pred.nonzero(as_tuple=True)] = sampled
            tokens = cur.clone()
            if cfg != 1.0:
                ctx_tok = ctx_tok[:bsz]
        return m.unpatchify(tokens)


# ── Data ──────────────────────────────────────────────────────────────────
class V5DepthDataset(Dataset):
    """v5_full: images/depth/*.{jpg,png} → targets/depth/{stem}.png"""

    def __init__(self, stems: list[str], img_map: dict[str, Path]):
        self.stems = stems
        self.img_map = img_map
        self.tgt_dir = Path(DATA_ROOT) / "targets" / "depth"

    def __len__(self):
        return len(self.stems)

    def _load_rgb(self, p: Path) -> torch.Tensor:
        im = Image.open(p).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 127.5 - 1.0).permute(2, 0, 1)

    def __getitem__(self, i):
        s = self.stems[i]
        return self._load_rgb(self.img_map[s]), self._load_rgb(self.tgt_dir / f"{s}.png")


def build_data():
    img_dir = Path(DATA_ROOT) / "images" / "depth"
    # Map stem→path for both jpg and png
    img_map: dict[str, Path] = {}
    for ext in ("*.jpg", "*.png"):
        for p in img_dir.glob(ext):
            img_map[p.stem] = p

    tgt_dir = Path(DATA_ROOT) / "targets" / "depth"
    tgts_avail = {p.stem for p in tgt_dir.glob("*.png")}
    # Only use stems with both image and target
    stems = sorted(set(img_map.keys()) & tgts_avail)
    log(f"data[depth] images={len(img_map)} targets_avail={len(tgts_avail)} usable_pairs={len(stems)}")

    rng = np.random.RandomState(SEED)
    rng.shuffle(stems)
    val_stems = stems[:N_VAL]
    train_stems = stems[N_VAL:N_VAL + N_TRAIN]

    train_ds = V5DepthDataset(train_stems, img_map)
    val_ds = V5DepthDataset(val_stems, img_map)
    log(f"  train={len(train_ds)} val={len(val_ds)}")

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=2, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)
    return train_dl, val_dl


# ── Train ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_latent(vae, imgs):
    return vae.encode(imgs).sample().mul_(VAE_SCALE)


def _interp_pos_grid(pe: torch.Tensor, new_hw: int) -> torch.Tensor:
    L, D = pe.shape[1], pe.shape[2]
    old_hw = int(round(L ** 0.5))
    g = pe.reshape(1, old_hw, old_hw, D).permute(0, 3, 1, 2)
    g = torch.nn.functional.interpolate(g, size=(new_hw, new_hw),
                                        mode="bilinear", align_corners=False)
    return g.permute(0, 2, 3, 1).reshape(1, new_hw * new_hw, D)


def resize_pos_keys(sd: dict, prefix: str, hw: int) -> dict:
    buf = 64
    for k in (prefix + "encoder_pos_embed_learned", prefix + "decoder_pos_embed_learned"):
        sd[k] = torch.cat([sd[k][:, :buf], _interp_pos_grid(sd[k][:, buf:], hw)], dim=1)
    sd[prefix + "diffusion_pos_embed_learned"] = _interp_pos_grid(sd[prefix + "diffusion_pos_embed_learned"], hw)
    return sd


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    log(f"device={DEVICE} steps={STEPS} batch={BATCH} lr={LR} out={OUT}")
    log(f"data_root={DATA_ROOT} img_size={IMG_SIZE} total_steps={TOTAL_STEPS}")

    base = mar_base(buffer_size=64, diffloss_d=6, diffloss_w=1024,
                    num_sampling_steps="100", img_size=IMG_SIZE)
    sd0 = load_file(f"{CKPT_DIR}/mar-base.safetensors")
    if IMG_SIZE != 256:
        sd0 = resize_pos_keys(sd0, "", IMG_SIZE // 16)
    base.load_state_dict(sd0, strict=True)
    model = MARVB(base).to(DEVICE).train()
    vae = AutoencoderKL(embed_dim=16, ch_mult=(1, 1, 2, 2, 4),
                        ckpt_path=f"{CKPT_DIR}/kl16.safetensors").to(DEVICE).eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    log(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M trainable")

    train_dl, val_dl = build_data()
    log(f"train_dl: {len(train_dl)} batches/epoch")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))

    def lr_at(step):
        if step < WARMUP:
            return LR * step / max(1, WARMUP)
        t = (step - WARMUP) / max(1, TOTAL_STEPS - WARMUP)
        return LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t)))

    @torch.no_grad()
    def val_loss():
        model.eval()
        losses = []
        for i, (im, tg) in enumerate(val_dl):
            if i >= 10:
                break
            im, tg = im.to(DEVICE), tg.to(DEVICE)
            losses.append(model(encode_latent(vae, im), encode_latent(vae, tg),
                                torch.zeros(im.size(0), dtype=torch.long, device=DEVICE)).item())
        model.train()
        return float(np.mean(losses))

    @torch.no_grad()
    def sample_panel(step):
        if DEVICE == "mps":
            torch.mps.empty_cache()
        model.eval()
        im, tg = next(iter(val_dl))
        im, tg = im[:4].to(DEVICE), tg[:4].to(DEVICE)
        ctx = encode_latent(vae, im)
        lat = model.sample_vb(ctx, torch.zeros(4, dtype=torch.long, device=DEVICE),
                              num_iter=SAMPLE_ITER, cfg=2.0)
        pred = vae.decode(lat / VAE_SCALE).clamp(-1, 1)
        mae = ((pred + 1) / 2 - (tg + 1) / 2).abs().mean().item()
        panel = torch.cat([im, tg, pred], dim=0)
        save_image(panel, OUT / f"panel_step{step}.png", nrow=4,
                   normalize=True, value_range=(-1, 1))
        log(f"[sample] step {step} rgb_mae={mae:.4f}")
        model.train()
        return mae

    best_val = float("inf")
    start_step = 0
    if RESUME and Path(RESUME).exists():
        sd = torch.load(RESUME, map_location="cpu", weights_only=False)
        msd = sd["model"]
        pos_resized = False
        if IMG_SIZE != 256 and msd["m.diffusion_pos_embed_learned"].shape[1] == 256:
            msd = resize_pos_keys(msd, "m.", IMG_SIZE // 16)
            pos_resized = True
        model.load_state_dict(msd)
        if "opt" in sd and not pos_resized:
            opt.load_state_dict(sd["opt"])
        start_step = sd["step"]
        best_val = sd.get("val_loss", best_val)
        log(f"[resume] {RESUME} step={start_step} val_loss={best_val:.4f}")

    train_iter = iter(train_dl)
    t0 = time.time()
    loss_hist: list[float] = []

    for step in range(start_step + 1, STEPS + 1):
        try:
            im, tg = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            im, tg = next(train_iter)
        im, tg = im.to(DEVICE), tg.to(DEVICE)
        labels = torch.zeros(im.size(0), dtype=torch.long, device=DEVICE)  # depth=0

        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        loss = model(encode_latent(vae, im), encode_latent(vae, tg), labels)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        loss_hist.append(loss.item())

        if step % 20 == 0:
            log(f"step {step}/{STEPS} loss={np.mean(loss_hist[-20:]):.4f} "
                f"lr={lr_at(step):.2e} gn={gn:.2f} ({(time.time()-t0)/max(1, step-start_step):.2f}s/it)")
        if step % 20 == 0:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "val_loss": best_val}, OUT / "ckpt_latest.pt")
        if step % VAL_EVERY == 0:
            vl = val_loss()
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                            "step": step, "val_loss": vl}, OUT / "ckpt_best.pt")
                log(f"[val] step {step} val_loss={vl:.4f} *BEST*")
            else:
                log(f"[val] step {step} val_loss={vl:.4f} best={best_val:.4f}")
        if step % SAMPLE_EVERY == 0 or step == STEPS:
            sample_panel(step)

    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "step": STEPS, "val_loss": best_val}, OUT / "ckpt_final.pt")
    json.dump({"best_val_loss": best_val, "steps": STEPS, "batch": BATCH, "lr": LR,
               "data": "vb_snvc_train_v5_full", "task": "depth"},
              open(OUT / "train_config.json", "w"), indent=2)
    log(f"[done] {(time.time()-t0)/60:.1f} min, best_val={best_val:.4f}")


if __name__ == "__main__":
    main()
