# Open-Vision-Banana 论文对齐深度估计训练数据技术报告

> **报告日期**: 2026-07-13  
> **目标论文**: Image Generators are Generalist Vision Learners (arXiv:2604.20329v3)  
> **任务类型**: `existing_data_analysis` — 基于现有数据基础设施的深度训练数据准备方案  
> **项目路径**: `/Users/gatilin/PycharmProjects/open-vision-banana`

---

## 1. 论文核心要求摘要

### 1.1 关键原文摘录

根据论文正文第 3.2 节（Metric Depth Estimation）及多处关键段落：

> *"To collect data for instruction tuning, we utilize in-house model annotations for web-crawled 2D images, as well as **synthetic data from rendering engines for 3D tasks**."*  
> — Section 2, Method

> *"Our model is trained **entirely on synthetic depth data created from simulation engines** — we use **zero real-world depth data**, and **exclude training data from any of the depth datasets we evaluate on**."*  
> — Section 3.2, Page 10-11

> *"Note that this result is achieved **without relying on camera parameters** (neither intrinsics nor extrinsics) during **both training or inference**."*  
> — Section 3.2, Page 11

### 1.2 训练数据约束清单

| 约束项 | 论文要求 | 当前项目状态 | 合规性 |
|--------|----------|-------------|--------|
| **深度数据来源** | 纯合成（simulation engines only） | 已有合成数据集注册；但存在伪标签脚本使用真实图像 | ⚠️ 需修正 |
| **真实深度数据** | 零使用（zero real-world depth data） | `distill_depth_pseudo_labels.py` 从真实图生成伪标签 | ❌ 不合规 |
| **评估数据泄漏** | 排除所有评估基准训练集 | `prepare_depth_dataset.py` 使用 COCO128（与评估无关，但非合成） | ⚠️ 需审查 |
| **相机参数** | 训练/推断均不使用内参/外参 | 现有代码不依赖相机参数 | ✅ 合规 |
| **数据混合比例** | 9:1 生成数据 : 视觉任务数据 | `MixedVisionDataset` + `vision_ratio=0.1` | ✅ 合规 |
| **深度编码** | Power transform + 多 colormap | `DepthRGBMapper` 完整实现 6 色图 | ✅ 合规 |
| **色图增强** | cube_edge / plasma / inferno / viridis / grayscale / rainbow | `mixed.py` 支持训练时动态色图切换 | ✅ 合规 |

---

## 2. 现有数据基础设施深度分析

### 2.1 数据集注册与下载系统

**文件**: `scripts/download_data.py`

当前 `DATASET_REGISTRY` 中注册的**合成深度数据集**（符合论文要求）：

```python
# scripts/download_data.py:59-109

DATASET_REGISTRY: dict[str, DatasetSpec] = {
    # === Paper-equivalent synthetic data (large) ===
    "hypersim-depth": DatasetSpec(
        name="Hypersim Depth",
        task="depth",
        source="https://github.com/apple/ml-hypersim",
        size_estimate_gb=1.5,
        requires_login=False,
        license="MIT",
    ),
    # === Lightweight alternatives to Hypersim ===
    "vkitti2-depth": DatasetSpec(
        name="Virtual KITTI 2 Depth",
        task="depth",
        source="https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/",
        size_estimate_gb=50.0,
        requires_login=False,
        license="Academic",
    ),
    "blendedmvs-depth": DatasetSpec(
        name="BlendedMVS Depth",
        task="depth",
        source="https://github.com/YoYo000/MVSNet",
        size_estimate_gb=17.0,
        requires_login=False,
        license="Academic",
    ),
    "hrwsi-depth": DatasetSpec(
        name="HRWSI Depth",
        task="depth",
        source="https://github.com/aim-uofa/DepthAnything",
        size_estimate_gb=20.0,
        requires_login=False,
        license="Academic",
    ),
    "tartanair-depth": DatasetSpec(
        name="TartanAir Depth",
        task="depth",
        source="https://github.com/castacks/tartanair_tools",
        size_estimate_gb=30.0,
        requires_login=False,
        license="Academic",
    ),
}
```

**分析**：项目已注册 5 个合成深度数据集，其中 **Hypersim**（室内合成）和 **Virtual KITTI 2**（室外合成）是最核心的两个，与论文"simulation engines"描述高度吻合。

### 2.2 Hypersim 数据准备管道

**文件**: `scripts/prepare_hypersim.py`

```python
# scripts/prepare_hypersim.py:74-127

def prepare_hypersim(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> int:
    """Write ``images/`` + ``targets/`` under *output*."""
    pairs = find_hypersim_pairs(source)
    # ...
    for idx, (color_path, depth_path) in enumerate(pairs):
        # Load color image
        color = Image.open(color_path).convert("RGB")
        if resolution is not None:
            color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # Load metric depth (meters) and apply power transform + cube-edge colormap
        depth = _load_depth_meters(depth_path)
        target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
        _tensor_to_pil_rgb(target_rgb).save(out_tgt)
        color.save(out_img)
```

**关键实现细节**：
- 深度值自动检测单位：若 `max > 1.0` 则除以 1000.0（mm → m）
- 深度 resize 使用 `nearest` 插值（保留深度不连续性）
- 输出格式：`images/` + `targets/` 平铺目录结构

### 2.3 深度 RGB 编解码器

**文件**: `src/open_vision_banana/core/depth_mapper.py`

论文公式实现（Section 3.2）：

```python
# src/open_vision_banana/core/depth_mapper.py:176-194

def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
    """Apply power transform with adaptive precision.

    f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)

    Args:
        depth: Metric depth values [B, 1, H, W] in meters, range [0, ∞).

    Returns:
        Normalized depth values [B, 1, H, W] in [0, 1).
    """
    depth = depth.clamp(min=0.0)
    ratio = depth / (self.lam * self.c)
    ratio = ratio.clamp(max=1.0 - 1e-7)
    base = 1.0 - ratio
    exponent = self.lam + 1  # = -2
    normalized = 1.0 - torch.pow(base, exponent)
    return normalized.clamp(0.0, 1.0)
```

参数默认值（论文标准）：`λ = -3.0`, `c = 10/3 ≈ 3.333`

支持的 6 种 colormap（论文 Section 3.2 及网站提及）：

```python
# src/open_vision_banana/core/depth_mapper.py:122-129

_COLORMAP_REGISTRY: dict[str, torch.Tensor] = {
    "cube_edge": CUBE_EDGE_KEYPOINTS,    # 论文默认: Black→Blue→Cyan→Green→Yellow→White
    "plasma": PLASMA_KEYPOINTS,          # matplotlib plasma
    "inferno": INFERNO_KEYPOINTS,        # matplotlib inferno
    "viridis": VIRIDIS_KEYPOINTS,        # matplotlib viridis
    "grayscale": GRAYSCALE_KEYPOINTS,    # 线性灰度
    "rainbow": RAINBOW_KEYPOINTS,        # 论文网站提及: black-red-yellow-green-cyan-blue-violet-white
}
```

### 2.4 数据集加载器

**文件**: `src/open_vision_banana/data/datasets.py`

已实现的合成/真实深度数据集加载器：

| 数据集 | 类型 | 是否合成 | 是否符合论文训练要求 |
|--------|------|---------|-------------------|
| `HypersimDepthDataset` | 室内深度 | ✅ 合成 | ✅ 可使用 |
| `NYUDepthDataset` | 室内深度 | ❌ 真实 | ❌ 仅用于评估（论文排除） |
| `DIODEDataset` | 室内+室外深度 | ❌ 真实 | ❌ 仅用于评估（论文排除） |
| `KITTIDataset` | 驾驶深度 | ❌ 真实（LiDAR） | ❌ 仅用于评估（论文排除） |
| `ETH3DDataset` | 高精度深度 | ❌ 真实（激光扫描） | ❌ 仅用于评估（论文排除） |
| `SintelFlowDataset` | 光流 | ✅ 合成动画 | ⚠️ 非深度任务 |

**注意**：`datasets.py` 中没有直接实现 Virtual KITTI 2 加载器，但 `HypersimDepthDataset` 的通用文件扫描逻辑可通过调整路径复用。

### 2.5 混合数据集（9:1 比例）

**文件**: `src/open_vision_banana/data/mixed.py`

```python
# src/open_vision_banana/data/mixed.py:52-76

class MixedVisionDataset(Dataset):
    def __init__(
        self,
        vision_datasets: list[Dataset],
        generation_dataset: Dataset | None = None,
        vision_ratio: float = 0.1,        # 论文: 9:1 = 0.1 vision
        seed: int = 42,
        colormaps: list[str] | None = None,
    ):
        # ...
        # Multi-colormap training augmentation (paper Section 3.2)
        self.colormaps = colormaps or ["cube_edge"]
```

训练时动态色图切换逻辑（`__getitem__` 内）：

```python
# src/open_vision_banana/data/mixed.py:107-122

if isinstance(sample, dict) and sample.get("task") == "depth":
    cmap_idx = int(torch.randint(len(self.colormaps), (1,), generator=self.rng).item())
    colormap = self.colormaps[cmap_idx]
    sample["colormap"] = colormap
    if colormap != "cube_edge" and "target_rgb" in sample:
        from ..core.depth_mapper import DepthRGBMapper
        mapper = DepthRGBMapper()
        target_rgb = sample["target_rgb"]
        depth = mapper.rgb_to_depth(target_rgb.unsqueeze(0)).squeeze(0)
        mapper.set_colormap(colormap)
        sample["target_rgb"] = mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
```

### 2.6 数据验证系统

**文件**: `src/open_vision_banana/data/validation.py`

```python
# src/open_vision_banana/data/validation.py:79-87

def validate_vision_dataset(
    data_path: str | Path,
    *,
    min_pairs: int = 1,
    require_square: bool = False,
    min_resolution: int = 64,
    max_resolution: int | None = None,
    check_value_range: bool = True,
) -> ValidationReport:
```

验证项包括：
- `images/` 和 `targets/` 目录存在性
- 配对文件一致性（stem 匹配）
- 图像尺寸一致性（image vs target）
- 分辨率范围检查
- 像素值范围检查
- 孤儿文件检测

---

## 3. 现有不合规数据识别与清理

### 3.1 不合规数据资产清单

以下数据目录/脚本**不满足**论文"纯合成深度数据"要求：

| 路径/脚本 | 问题描述 | 风险等级 |
|-----------|---------|---------|
| `scripts/distill_depth_pseudo_labels.py` | 使用 Depth-Anything V2 从**真实图像**生成伪深度标签 | 🔴 高 |
| `scripts/prepare_depth_dataset.py` | 使用 DPT 从 COCO128 **真实图像**生成深度 | 🔴 高 |
| `data/vb_depth_pseudo_5k/` | 由 `distill_depth_pseudo_labels.py` 生成的伪标签数据集 | 🔴 高 |
| `data/vision_banana_depth/` | 由 `prepare_depth_dataset.py` 生成的 COCO 深度数据 | 🔴 高 |
| `configs/training_mps_depth_aligned.yaml` | 配置引用了 `vb_depth_pseudo_5k` | 🟡 中 |

### 3.2 清理命令

```bash
# 1. 确认不合规数据目录大小
du -sh data/vb_depth_pseudo_5k/ data/vision_banana_depth/ data/vb_distill_428/

# 2. 备份后删除（建议备份到外部存储）
mkdir -p /tmp/ovb_noncompliant_backup
cp -r data/vb_depth_pseudo_5k /tmp/ovb_noncompliant_backup/
cp -r data/vision_banana_depth /tmp/ovb_noncompliant_backup/
rm -rf data/vb_depth_pseudo_5k/ data/vision_banana_depth/

# 3. 标记伪标签脚本为"非论文对齐"（防止误用）
mv scripts/distill_depth_pseudo_labels.py scripts/ARCHIVE_distill_depth_pseudo_labels.py
mv scripts/prepare_depth_dataset.py scripts/ARCHIVE_prepare_depth_dataset.py
```

---

## 4. 论文对齐的纯合成深度数据准备方案

### 4.1 推荐合成数据集组合

基于论文"entirely synthetic depth data from simulation engines"要求，建议采用以下**纯合成**数据集组合：

| 数据集 | 场景类型 | 规模估计 | 用途 | 下载/准备状态 |
|--------|---------|---------|------|-------------|
| **Hypersim** | 合成室内 | ~77K frames, 1.5GB | 主要训练数据 | 项目已有准备脚本 |
| **Virtual KITTI 2** | 合成驾驶 | ~50GB | 室外/驾驶场景补充 | 项目已注册，需自定义准备 |
| **BlendedMVS** (可选) | 合成多视角 | ~17GB | 增加视角多样性 | 项目已注册 |
| **TartanAir** (可选) | 合成室内/室外导航 | ~30GB | 增加场景多样性 | 项目已注册 |

**核心原则**：
1. **仅使用合成数据集**
2. **排除 NYU, DIODE, KITTI, ETH3D, iBims1, nuScenes 的训练集**（这些是评估基准）
3. **不使用相机内参/外参**

### 4.2 Hypersim 数据准备（主要数据源）

Hypersim 是 Apple 发布的合成室内场景数据集，使用 V-Ray 渲染引擎生成，包含精确的 metric depth ground truth。

#### 步骤 1: 下载 Hypersim

```bash
# Hypersim 使用公开 AWS S3 存储，无需登录
# 参考: https://github.com/apple/ml-hypersim

# 创建下载目录
mkdir -p data/hypersim_raw
cd data/hypersim_raw

# 使用 AWS CLI 下载（推荐）
# 仅下载 camera_*.hdf5 和 rendering 目录下的 color/depth 对
aws s3 cp --recursive s3://hypersim/evermotion_dataset/ .

# 或使用官方提供的下载脚本
# git clone https://github.com/apple/ml-hypersim.git
# cd ml-hypersim
# python code/python/tools/dataset_download.py --downloads_dir ../../hypersim_raw
```

#### 步骤 2: 使用项目脚本准备 Hypersim

```bash
# 使用项目提供的 prepare_hypersim.py
cd /Users/gatilin/PycharmProjects/open-vision-banana

python scripts/prepare_hypersim.py \
  --source data/hypersim_raw \
  --output data/hypersim_depth_paper_aligned/ \
  --resolution 512 \
  --validate
```

**脚本内部关键流程**（来自 `scripts/prepare_hypersim.py:74-127`）：

```python
def prepare_hypersim(source, output, *, max_samples=None, resolution=None, overwrite=False):
    pairs = find_hypersim_pairs(source)
    # 配对逻辑: *color.png ↔ *depth.png
    
    mapper = DepthRGBMapper()  # 默认: cube_edge, lam=-3.0, c=10/3
    
    for color_path, depth_path in pairs:
        color = Image.open(color_path).convert("RGB")
        if resolution:
            color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        depth = _load_depth_meters(depth_path)  # 自动 mm→m 转换
        if resolution:
            depth = F.interpolate(depth, size=(resolution, resolution), mode="nearest")
        
        # 应用论文 power transform + cube-edge colormap
        target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
        # 保存到 images/ + targets/
```

#### 步骤 3: 验证准备结果

```bash
# 使用项目验证脚本
python scripts/validate_dataset.py \
  --data-path data/hypersim_depth_paper_aligned/ \
  --task depth

# 或使用 Python API
python -c "
from open_vision_banana.data.validation import validate_vision_dataset
report = validate_vision_dataset('data/hypersim_depth_paper_aligned/')
print(report.summary())
assert report.ok, 'Validation failed!'
"
```

### 4.3 Virtual KITTI 2 数据准备（室外补充）

Virtual KITTI 2 使用 Unity 游戏引擎渲染，包含精确的 metric depth。

#### 步骤 1: 下载

```bash
# 官网: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
# 需要填写学术使用协议表格后获取下载链接

mkdir -p data/vkitti2_raw
cd data/vkitti2_raw

# 下载后解压，预期目录结构:
# vkitti2_raw/
#   Scene01/
#     15-deg-left/
#       frames/rgb/Camera_0/        ← RGB 图像
#       frames/depth/Camera_0/      ← 深度图 (PNG, meters × 100)
```

#### 步骤 2: 自定义准备脚本

项目目前没有内置的 vkitti2 准备脚本，需要新建：

```python
#!/usr/bin/env python3
"""Prepare Virtual KITTI 2 depth data for Vision Banana training.

VKITTI2 depth format:
  - PNG files with depth values in cm (divide by 100 for meters)
  - RGB files: *.jpg
  - Directory: SceneXX/variant/frames/rgb/Camera_0/  and  depth/Camera_0/
"""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from open_vision_banana.core.depth_mapper import DepthRGBMapper


def prepare_vkitti2(source: Path, output: Path, resolution: int = 512):
    img_dir = output / "images"
    tgt_dir = output / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    
    mapper = DepthRGBMapper()
    pairs = []
    
    # 递归查找 RGB/Depth 对
    for rgb_path in sorted(source.rglob("**/frames/rgb/Camera_0/*.jpg")):
        depth_path = Path(str(rgb_path).replace("/rgb/", "/depth/").replace(".jpg", ".png"))
        if depth_path.exists():
            pairs.append((rgb_path, depth_path))
    
    print(f"Found {len(pairs)} VKITTI2 RGB-D pairs")
    
    for idx, (rgb_path, depth_path) in enumerate(pairs):
        # Load RGB
        rgb = Image.open(rgb_path).convert("RGB")
        if resolution:
            rgb = rgb.resize((resolution, resolution), Image.Resampling.LANCZOS)
        
        # Load depth (VKITTI2: cm → m)
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img).astype(np.float32) / 100.0  # cm → meters
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        
        if resolution:
            depth = torch.nn.functional.interpolate(
                depth, size=(resolution, resolution), mode="nearest"
            )
        
        # Encode to RGB visualization
        target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
        arr = (target_rgb.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        
        # Save
        stem = f"{idx:06d}_vkitti2"
        rgb.save(img_dir / f"{stem}.png")
        Image.fromarray(arr, mode="RGB").save(tgt_dir / f"{stem}.png")
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} / {len(pairs)}")
    
    print(f"Wrote {len(pairs)} pairs to {output}")
    return len(pairs)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default="data/vkitti2_depth_paper_aligned")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    prepare_vkitti2(args.source, args.output, args.resolution)
```

保存为 `scripts/prepare_vkitti2.py` 并执行：

```bash
python scripts/prepare_vkitti2.py \
  --source data/vkitti2_raw \
  --output data/vkitti2_depth_paper_aligned/ \
  --resolution 512
```

### 4.4 训练配置（论文对齐）

新建配置文件 `configs/training_depth_paper_aligned.yaml`：

```yaml
# configs/training_depth_paper_aligned.yaml
# Vision Banana 深度估计训练配置 — 严格对齐 arXiv:2604.20329v3
# 核心约束: entirely synthetic depth data, zero real-world depth, no camera intrinsics

model:
  name: "black-forest-labs/FLUX.1-schnell"
  dtype: "bfloat16"

lora:
  rank: 64
  alpha: 128
  rs_lora: true
  target_modules:
    - "to_k"
    - "to_v"
  dropout: 0.0

optimizer:
  lr: 5.0e-5
  weight_decay: 0.01
  scheduler: "cosine"
  warmup_steps: 500
  max_grad_norm: 1.0

output_dir: "output/checkpoints_depth_paper_aligned"

logging:
  wandb_enabled: false
  tensorboard_enabled: true
  tensorboard_dir: "output/runs_depth_paper_aligned"
  run_name: "depth_paper_aligned"

training:
  epochs: 10
  gradient_accumulation_steps: 1
  use_amp: true
  use_ema: true
  ema_decay: 0.9999
  ema_warmup_steps: 2000
  save_every: 1000
  eval_every: 500
  max_keep: 5
  seed: 42
  log_every: 10

data:
  # 论文: "mix vision task data at a very low ratio"
  # 9:1 generation:vision = 0.1 vision_ratio
  vision_ratio: 0.1
  generation_ratio: 0.9
  resolution: 512
  batch_size: 8
  num_workers: 4
  augmentation: true
  val_split: 0.0
  generation_path: "data/vb_generation_5k"  # 保持生成能力的数据

  # 论文: 多 colormap 训练增强
  colormaps:
    - "cube_edge"
    - "plasma"
    - "inferno"
    - "viridis"
    - "grayscale"

  tasks:
    depth_estimation:
      datasets:
        # === 纯合成深度数据（论文对齐）===
        - name: "Hypersim"
          split: "train"
          path: "data/hypersim_depth_paper_aligned"
        - name: "Virtual_KITTI_2"
          split: "train"
          path: "data/vkitti2_depth_paper_aligned"
        # 可选补充:
        # - name: "BlendedMVS"
        #   split: "train"
        # - name: "TartanAir"
        #   split: "train"

depth_mapping:
  lam: -3.0
  c: 3.333
```

### 4.5 启动训练

```bash
# 使用统一启动器
bash scripts/unified_launch.sh \
  --config configs/training_depth_paper_aligned.yaml \
  --mode ddp

# 或单卡训练
python scripts/train.py \
  --config configs/training_depth_paper_aligned.yaml
```

---

## 5. 零样本评估基准（论文对齐）

论文在以下**真实世界基准**上进行零样本评估（训练时排除这些数据）：

| 基准 | 类型 | 论文指标 |
|------|------|---------|
| NYU Depth v2 | 室内真实 | δ1↑, AbsRel↓ |
| iBims1 | 室内真实 | δ1↑, AbsRel↓ |
| ETH3D | 室内+室外真实（激光扫描） | δ1↑, AbsRel↓ |
| DIODE-Indoor | 室内真实 | δ1↑, AbsRel↓ |
| KITTI | 驾驶真实（LiDAR） | δ1↑, AbsRel↓ |
| nuScenes | 驾驶真实 | δ1↑, AbsRel↓ |

评估命令：

```bash
# 评估 NYU Depth v2
python scripts/evaluate.py \
  --task depth \
  --checkpoint output/checkpoints_depth_paper_aligned/best.pt \
  --dataset nyu \
  --data-path /data/nyu_depth_v2/

# 评估 KITTI
python scripts/evaluate.py \
  --task depth \
  --checkpoint output/checkpoints_depth_paper_aligned/best.pt \
  --dataset kitti \
  --data-path /data/kitti/

# 评估 ETH3D
python scripts/evaluate.py \
  --task depth \
  --checkpoint output/checkpoints_depth_paper_aligned/best.pt \
  --dataset eth3d \
  --data-path /data/eth3d/
```

---

## 6. 代码审查清单

在提交训练前，使用以下清单确认数据合规性：

```python
# scripts/verify_paper_alignment.py
"""验证深度训练数据是否严格对齐论文要求."""

from pathlib import Path
import json

def verify_synthetic_only(data_root: Path) -> dict:
    """检查数据目录中是否仅包含合成来源的数据."""
    issues = []
    
    # 检查已知不合规目录名
    forbidden_names = ["nyu", "kitti", "diode", "eth3d", "ibims", "nuscenes",
                       "pseudo", "distill", "coco", "real"]
    for subdir in data_root.iterdir():
        if any(f in subdir.name.lower() for f in forbidden_names):
            issues.append(f"Potential non-synthetic data: {subdir.name}")
    
    # 检查 metadata 中是否有合成来源标记
    meta_files = list(data_root.rglob("metadata.json"))
    synthetic_confirmed = False
    for mf in meta_files:
        try:
            with open(mf) as f:
                meta = json.load(f)
            source = meta.get("source", "")
            if "hypersim" in source.lower() or "vkitti" in source.lower() or "synthetic" in source.lower():
                synthetic_confirmed = True
        except Exception:
            pass
    
    return {
        "synthetic_confirmed": synthetic_confirmed,
        "issues": issues,
        "compliant": len(issues) == 0 and synthetic_confirmed,
    }

if __name__ == "__main__":
    result = verify_paper_alignment(Path("data/hypersim_depth_paper_aligned"))
    print(json.dumps(result, indent=2))
    assert result["compliant"], "Data does NOT comply with paper requirements!"
```

---

## 7. 总结与行动项

### 7.1 项目基础设施成熟度评估

| 模块 | 成熟度 | 备注 |
|------|--------|------|
| 数据集注册/下载 | ✅ 高 | 5 个合成深度数据集已注册 |
| Hypersim 准备脚本 | ✅ 高 | `prepare_hypersim.py` 完整可用 |
| 深度 RGB 编解码 | ✅ 高 | `DepthRGBMapper` 6 色图完整实现 |
| 混合数据集 | ✅ 高 | `MixedVisionDataset` 支持 9:1 + 多色图 |
| 数据验证 | ✅ 高 | `validate_vision_dataset` 全面检查 |
| VKITTI2 准备脚本 | ⚠️ 缺失 | 需新建 `scripts/prepare_vkitti2.py` |
| 伪标签脚本清理 | ❌ 待办 | 需归档/移除不合规脚本 |
| 论文对齐配置 | ⚠️ 待创建 | 需新建 `training_depth_paper_aligned.yaml` |

### 7.2 优先行动项

1. **立即执行**: 归档 `distill_depth_pseudo_labels.py` 和 `prepare_depth_dataset.py`
2. **立即执行**: 清理 `data/vb_depth_pseudo_5k/` 和 `data/vision_banana_depth/`
3. **高优先级**: 下载并准备 Hypersim 数据集
4. **高优先级**: 创建 `scripts/prepare_vkitti2.py` 并准备 VKITTI2 数据
5. **中优先级**: 创建 `configs/training_depth_paper_aligned.yaml`
6. **中优先级**: 运行验证脚本确认数据合规性
7. **低优先级**: 可选添加 BlendedMVS / TartanAir 增强多样性

---

## 附录 A: 关键文件索引

| 文件路径 | 作用 | 相关行号 |
|---------|------|---------|
| `scripts/download_data.py` | 数据集注册与下载入口 | 59-143 |
| `scripts/prepare_hypersim.py` | Hypersim 数据准备 | 1-200 |
| `src/open_vision_banana/core/depth_mapper.py` | 深度↔RGB 编解码 | 1-357 |
| `src/open_vision_banana/data/datasets.py` | 数据集加载器集合 | 1-1783 |
| `src/open_vision_banana/data/mixed.py` | 9:1 混合 + 多色图增强 | 1-139 |
| `src/open_vision_banana/data/validation.py` | 数据完整性验证 | 1-265 |
| `configs/unified_training_config.yaml` | 统一训练配置模板 | 1-146 |
| `configs/training_mps_depth_aligned.yaml` | 现有 MPS 深度配置（引用伪标签数据） | 1-86 |

## 附录 B: 论文关键页码参考

| 内容 | 页码 | 关键信息 |
|------|------|---------|
| 摘要/引言 | 1-3 | 模型架构与任务概述 |
| Method (Section 2) | 4 | "synthetic data from rendering engines for 3D tasks" |
| Metric Depth Estimation (Section 3.2) | 9-12 | 深度估计方法、power transform 公式、纯合成数据声明 |
| 深度评估结果 (Table 6) | 11 | 6 个基准上的零样本结果 |
| 参考文献 | 16-23 | 相关数据集引用 |
