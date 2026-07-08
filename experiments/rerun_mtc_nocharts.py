#!/usr/bin/env python3
"""Re-run Multi-Task Consistency experiment — NO matplotlib, JSON+report only.

Fast vectorized ray tracing + valid-mask H4. Skips chart generation to avoid
matplotlib font-cache build issues in constrained environments.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from open_vision_banana.core.depth_mapper import DepthRGBMapper
from open_vision_banana.core.normal_mapper import NormalRGBMapper
from open_vision_banana.core.segmentation import SegmentationDecoder

MAX_DEPTH = 8.0
RESOLUTION = 128


# ── Vectorized Ray Tracing ───────────────────────────────────────
def _sphere_hit(origins, dirs, center, radius):
    oc = origins - center[None, :]
    a = np.sum(dirs * dirs, axis=1)
    b = 2.0 * np.sum(oc * dirs, axis=1)
    c = np.sum(oc * oc, axis=1) - radius ** 2
    disc = b * b - 4 * a * c
    hit = disc >= 0
    sqrt_d = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sqrt_d) / (2 * a)
    t2 = (-b + sqrt_d) / (2 * a)
    t_cand = np.where(t1 > 1e-4, t1, t2)
    valid = hit & (t_cand > 1e-4)
    t = np.full(len(dirs), np.inf); t[valid] = t_cand[valid]
    pts = origins + t[:, None] * dirs
    normals = (pts - center[None, :]) / radius
    return t, normals, valid


def _plane_hit(origins, dirs, point, normal_vec):
    denom = np.sum(dirs * normal_vec[None, :], axis=1)
    valid = np.abs(denom) > 1e-6
    diff = point[None, :] - origins
    t_cand = np.sum(diff * normal_vec[None, :], axis=1) / np.where(np.abs(denom) > 1e-6, denom, 1.0)
    good = valid & (t_cand > 1e-4)
    t = np.full(len(dirs), np.inf); t[good] = t_cand[good]
    normals = np.broadcast_to(normal_vec[None, :], (len(dirs), 3)).copy()
    return t, normals, good


def _cylinder_hit(origins, dirs, center_xz, radius):
    ox = origins[:, 0] - center_xz[0]
    oz = origins[:, 2] - center_xz[1]
    dx, dz = dirs[:, 0], dirs[:, 2]
    a = dx * dx + dz * dz
    va = np.abs(a) > 1e-8
    b = 2 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - radius ** 2
    disc = b * b - 4 * a * c
    hit = va & (disc >= 0)
    sqrt_d = np.sqrt(np.maximum(disc, 0))
    a_s = np.where(va, a, 1.0)
    t1 = (-b - sqrt_d) / (2 * a_s)
    t2 = (-b + sqrt_d) / (2 * a_s)
    t_cand = np.where(t1 > 1e-4, t1, t2)
    valid = hit & (t_cand > 1e-4)
    t = np.full(len(dirs), np.inf); t[valid] = t_cand[valid]
    pts = origins + t[:, None] * dirs
    normals = np.zeros_like(origins)
    normals[:, 0] = pts[:, 0] - center_xz[0]
    normals[:, 2] = pts[:, 2] - center_xz[1]
    n_len = np.sqrt(normals[:, 0] ** 2 + normals[:, 2] ** 2).clip(1e-8)
    normals /= n_len[:, None]
    return t, normals, valid


def render_scene(objects, resolution=128):
    H = W = resolution
    tan_fov = np.tan(np.radians(60.0) / 2.0)
    ys, xs = np.arange(H), np.arange(W)
    xx, yy = np.meshgrid(xs, ys)
    u = (2.0 * (xx + 0.5) / W - 1.0) * tan_fov
    v = (1.0 - 2.0 * (yy + 0.5) / H) * tan_fov
    dirs = np.stack([u, v, np.ones_like(u)], axis=-1).reshape(-1, 3)
    dirs /= np.sqrt(np.sum(dirs ** 2, axis=1, keepdims=True))
    origins = np.zeros_like(dirs)
    N = len(dirs)
    best_t = np.full(N, np.inf)
    best_n = np.zeros((N, 3)); best_n[:, 2] = 1.0
    best_oid = np.zeros(N, dtype=np.int64)

    for otype, params, oid in objects:
        if otype == "sphere":
            t, n, v = _sphere_hit(origins, dirs, params["center"], params["radius"])
        elif otype == "plane":
            t, n, v = _plane_hit(origins, dirs, params["point"], params["normal"])
        elif otype == "cylinder":
            t, n, v = _cylinder_hit(origins, dirs, params["center_xz"], params["radius"])
        else:
            continue
        closer = v & (t < best_t)
        best_t[closer] = t[closer]
        best_n[closer] = n[closer]
        best_oid[closer] = oid

    hit_ok = (best_t < np.inf) & (best_t <= MAX_DEPTH)
    best_t[~hit_ok] = MAX_DEPTH
    best_n[~hit_ok] = np.array([0.0, 0.0, 1.0])
    best_oid[~hit_ok] = 0
    positions = origins + best_t[:, None] * dirs

    return {
        "depth": torch.from_numpy(best_t.reshape(1, 1, H, W).astype(np.float32)),
        "normal": torch.from_numpy(best_n.reshape(H, W, 3).transpose(2, 0, 1)[None].astype(np.float32)),
        "position": torch.from_numpy(positions.reshape(H, W, 3).transpose(2, 0, 1)[None].astype(np.float32)),
        "object_id": torch.from_numpy(best_oid.reshape(1, 1, H, W)),
        "valid_mask": torch.from_numpy(hit_ok.reshape(1, 1, H, W).astype(np.float32)),
    }


# ── Scenes ───────────────────────────────────────────────────────
SCENE_A = [
    ("plane", {"point": np.array([0, -0.5, 0], dtype=np.float32),
               "normal": np.array([0, 1, 0], dtype=np.float32)}, 1),
    ("sphere", {"center": np.array([0, 0, 2.5], dtype=np.float32), "radius": 0.5}, 2),
    ("cylinder", {"center_xz": np.array([1.0, 2.5], dtype=np.float32), "radius": 0.25}, 3),
]
SCENE_B = [
    ("plane", {"point": np.array([0, -0.5, 0], dtype=np.float32),
               "normal": np.array([0, 1, 0], dtype=np.float32)}, 1),
    ("sphere", {"center": np.array([-0.4, 0.1, 2.2], dtype=np.float32), "radius": 0.5}, 2),
    ("sphere", {"center": np.array([0.4, 0.1, 2.0], dtype=np.float32), "radius": 0.4}, 3),
]
sc_n = np.array([0.2, 1.0, 0.1], dtype=np.float32); sc_n /= np.linalg.norm(sc_n)
SCENE_C = [
    ("plane", {"point": np.array([0, -0.3, 0], dtype=np.float32), "normal": sc_n}, 1),
    ("sphere", {"center": np.array([0, 0.4, 2.5], dtype=np.float32), "radius": 0.4}, 2),
]
SCENES = {
    "Scene A (Ground+Sphere+Cyl)": SCENE_A,
    "Scene B (Overlapping Spheres)": SCENE_B,
    "Scene C (Tilted Plane)": SCENE_C,
}


# ── Experiment Functions ─────────────────────────────────────────
def run_h1(depth, normal, position, valid_mask, name):
    pos = position[0]
    pr = torch.cat([pos[:, :, 1:], pos[:, :, -1:]], dim=2)
    pd = torch.cat([pos[:, 1:, :], pos[:, -1:, :]], dim=1)
    vr, vd = pr - pos, pd - pos
    nx = vr[1]*vd[2]-vr[2]*vd[1]; ny = vr[2]*vd[0]-vr[0]*vd[2]; nz = vr[0]*vd[1]-vr[1]*vd[0]
    nfd = torch.stack([nx, ny, nz]); nfd = nfd / nfd.norm(dim=0, keepdim=True).clamp(min=1e-8)
    dot = (nfd * normal[0]).sum(dim=0).clamp(-1, 1)
    aerr = torch.acos(dot) * 180.0 / np.pi
    v = valid_mask[0, 0] > 0.5
    ma = aerr[v].mean().item() if v.any() else 90.0

    sx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
    sy = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
    gx = F.conv2d(depth, sx, padding=1); gy = F.conv2d(depth, sy, padding=1)
    gm = torch.sqrt(gx[0,0]**2 + gy[0,0]**2 + 1e-10)
    nm = torch.sqrt(normal[0,0]**2 + normal[0,1]**2)
    if v.sum() > 10:
        g, n = gm[v].reshape(-1), nm[v].reshape(-1)
        corr = ((g-g.mean())*(n-n.mean())).sum() / (((g-g.mean())**2).sum().sqrt()*((n-n.mean())**2).sum().sqrt()+1e-10)
    else:
        corr = torch.tensor(0.0)
    return {"scene": name, "angular_error_deg": ma, "gradient_correlation": corr.item()}


def run_h2(depth, object_map, name):
    sx = torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]], dtype=torch.float32)
    sy = torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]], dtype=torch.float32)
    gx = F.conv2d(depth, sx, padding=1); gy = F.conv2d(depth, sy, padding=1)
    gm = torch.sqrt(gx**2 + gy**2)
    de = (gm > (gm.mean() + 2*gm.std())).float()
    sex = (object_map[:,:,:,:-1] != object_map[:,:,:,1:]).float()
    sey = (object_map[:,:,:-1,:] != object_map[:,:,1:,:]).float()
    sex = F.pad(sex, (0,1,0,0)); sey = F.pad(sey, (0,0,0,1))
    se = ((sex+sey)>0).float()
    iou = ((de*se).sum() / ((de+se)>0).float().sum().clamp(min=1)).item()
    return {"scene": name, "boundary_iou": iou, "depth_edge_ratio": de.mean().item(),
            "seg_boundary_ratio": se.mean().item()}


def run_h3(depth_rgb, normal_rgb, seg_rgb):
    def hm(x): return tuple(x[0].mean(dim=(1,2)).tolist())
    def hi(a, b, bins=16):
        ah = torch.histc(a, bins=bins, min=0, max=1)
        bh = torch.histc(b, bins=bins, min=0, max=1)
        return torch.min(ah/ah.sum(), bh/bh.sum()).sum().item()
    d, n, s = depth_rgb.reshape(-1), normal_rgb.reshape(-1), seg_rgb.reshape(-1)
    return {"depth_mean_rgb": list(hm(depth_rgb)), "normal_mean_rgb": list(hm(normal_rgb)),
            "seg_mean_rgb": list(hm(seg_rgb)),
            "histogram_intersections": {"depth_vs_normal": hi(d,n), "depth_vs_segmentation": hi(d,s),
                                         "normal_vs_segmentation": hi(n,s)}}


def run_h4(depth, normal, seg_rgb, valid_mask, noise_levels):
    dm = DepthRGBMapper(); nm = NormalRGBMapper(); dec = SegmentationDecoder()
    drgb = dm.depth_to_rgb(depth); nrgb = nm.normal_to_rgb(normal)
    mask = valid_mask > 0.5
    results = []
    for sigma in noise_levels:
        dn = (drgb + torch.randn_like(drgb)*sigma).clamp(0,1)
        dr = dm.rgb_to_depth(dn); err = (depth-dr).abs()
        mae = err[mask.expand_as(err)].mean().item() if mask.any() else err.mean().item()
        nr = nm.rgb_to_normal(nrgb)
        cs = F.cosine_similarity(normal, nr, dim=1).mean().item()
        sn = (seg_rgb + torch.randn_like(seg_rgb)*sigma).clamp(0,1)
        inst = dec.decode_instance(sn, adaptive_color_tolerance=False)
        ni = len(inst[0]) if inst else 0; exp = 3
        f1 = 2*min(ni,exp)/(ni+exp) if (ni+exp)>0 else 0
        results.append({"noise_sigma": sigma, "depth_roundtrip_mae": mae,
                         "normal_roundtrip_cos_sim": cs, "seg_f1_proxy": f1})
    return results


# ── Main ─────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Multi-Task Consistency — FIXED RERUN (no charts)")
    print("=" * 60)

    t0 = time.perf_counter()
    gts = {}
    for name, objs in SCENES.items():
        gts[name] = render_scene(objs, RESOLUTION)
        d = gts[name]["depth"]; v = gts[name]["valid_mask"]
        print(f"  {name}: depth [{d.min():.2f},{d.max():.2f}]m  valid={v.sum().int().item()}/{v.numel()}")
    print(f"  Render: {time.perf_counter()-t0:.2f}s\n")

    dm, nm = DepthRGBMapper(), NormalRGBMapper()
    ga = gts["Scene A (Ground+Sphere+Cyl)"]

    drgb = dm.depth_to_rgb(ga["depth"]); nrgb = nm.normal_to_rgb(ga["normal"])
    seg = torch.zeros_like(ga["depth"]).expand(-1,3,-1,-1).clone()
    for oid in ga["object_id"].unique().tolist():
        m = (ga["object_id"]==oid).float()
        c = torch.tensor([.2+.3*(oid%3),.2+.3*((oid+1)%3),.2+.3*((oid+2)%3)]).view(1,3,1,1)
        seg += m*c
    seg = seg.clamp(0,1)

    h3 = run_h3(drgb, nrgb, seg)
    print(f"[H3] max overlap: {max(h3['histogram_intersections'].values()):.3f}")

    h4 = run_h4(ga["depth"], ga["normal"], seg, ga["valid_mask"], [0.0, 0.05, 0.1, 0.2, 0.5])
    for r in h4:
        print(f"[H4] σ={r['noise_sigma']:.2f}: MAE={r['depth_roundtrip_mae']:.4f}m  cos={r['normal_roundtrip_cos_sim']:.6f}  F1={r['seg_f1_proxy']:.3f}")

    h1s, h2s = [], []
    for name in SCENES:
        g = gts[name]
        r1 = run_h1(g["depth"], g["normal"], g["position"], g["valid_mask"], name)
        r2 = run_h2(g["depth"], g["object_id"], name)
        print(f"[{name}] H1 ang={r1['angular_error_deg']:.2f}°  H2 IoU={r2['boundary_iou']:.3f}")
        h1s.append(r1); h2s.append(r2)

    out = Path(__file__).parent
    data = {
        "metadata": {"run_date": date.today().isoformat(), "resolution": RESOLUTION,
                      "max_depth_m": MAX_DEPTH, "valid_mask_filtering": True,
                      "note": "Rerun with vectorized ray tracer + valid-mask H4"},
        "h1": h1s, "h2": h2s, "h3": h3, "h4": h4,
    }
    p = out / "multi_task_consistency_results.json"
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nSaved: {p}")

    # Report
    rp = out / "multi_task_consistency_report.md"
    md = f"""# Multi-Task Spatial Consistency Experiment Report (Fixed Rerun)

> **Date**: {date.today().isoformat()} | **Resolution**: {RESOLUTION}² | **Max depth**: {MAX_DEPTH}m
> **Key fix**: Valid-mask filtering in H4 eliminates σ=0 MAE=6177m artifact

## H1: Depth-Normal Geometric Consistency

| Scene | Angular Error (°) | Grad Correlation |
|-------|-------------------|------------------|
"""
    for r in h1s: md += f"| {r['scene']} | {r['angular_error_deg']:.2f} | {r['gradient_correlation']:.3f} |\n"
    md += "\n## H2: Segmentation-Depth Boundary Alignment\n\n| Scene | IoU | Depth Edge | Seg Boundary |\n|-------|-----|------------|-------------|\n"
    for r in h2s: md += f"| {r['scene']} | {r['boundary_iou']:.3f} | {r['depth_edge_ratio']:.4f} | {r['seg_boundary_ratio']:.4f} |\n"
    md += f"\n## H3: RGB Encoding Independence\n\nMax histogram overlap: {max(h3['histogram_intersections'].values()):.3f}\n"
    md += "\n## H4: Noise Isolation (valid-mask filtered)\n\n| σ | Depth MAE (m) | Normal CosSim | Seg F1 |\n|---|---------------|---------------|--------|\n"
    for r in h4: md += f"| {r['noise_sigma']:.2f} | {r['depth_roundtrip_mae']:.4f} | {r['normal_roundtrip_cos_sim']:.6f} | {r['seg_f1_proxy']:.3f} |\n"
    md += f"\n**σ=0 MAE is now {h4[0]['depth_roundtrip_mae']:.4f}m** (was 6177m before valid-mask fix).\n\n---\n*Generated by rerun_mtc_nocharts.py*\n"
    rp.write_text(md, encoding="utf-8")
    print(f"Saved: {rp}")
    print("\nDONE")


if __name__ == "__main__":
    main()
