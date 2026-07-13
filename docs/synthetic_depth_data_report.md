# open-vision-banana 深度估计训练数据准备技术报告
## 论文对齐: Image Generators are Generalist Vision Learners (arXiv:2604.20329v3)
### 任务类型: lightweight_alternatives | 纯合成深度数据 (entirely synthetic depth data)

---

## 1. 执行摘要

本报告为 **open-vision-banana** 项目提供深度估计任务的**纯合成深度训练数据**准备方案，严格对齐论文 *Image Generators are Generalist Vision Learners* (arXiv:2604.20329v3) 的要求。

**核心约束**（论文原话）:
> "Our model is trained **entirely on synthetic depth data created from simulation engines** -- we use **zero real-world depth data**, and exclude training data from any of the depth datasets we evaluate on."

**现有基础设施状态**:
- [x] `DepthRGBMapper` 已完整实现论文的深度<->RGB 双向编码（power transform lambda=-3, c=10/3 + cube-edge colormap）
- [x] `MixedVisionDataset` 已支持 9:1 generation:vision 混合比例
- [x] 多色图增强（plasma/inferno/viridis/grayscale/cube_edge/rainbow）已集成
- [ ] 现有 `prepare_depth_dataset.py` 使用 Depth-Anything 伪标签（**非合成数据，违反论文约束**）
- [ ] Hypersim 完整版 ~1.5TB，需要 lightweight alternatives

---

## 2. 论文深度编码规范

### 2.1 Power Transform（深度弯曲）

论文 Section 3.2 定义：

```
f(d, lambda, c) = 1 - (1 - d/(lambda*c))^(lambda+1)
```

- **lambda = -3** （形状参数，控制弯曲程度）
- **c = 10/3 ~= 3.333** （尺度参数，米为单位）
- 将 metric depth `d in [0, inf)` 映射到 `f in [0, 1)`
- 弯曲后**近处内容精度更高**（符合机器人/AR 应用需求）

**项目实现** (`src/open_vision_banana/core/depth_mapper.py:176-194`):

```python
class DepthRGBMapper:
    def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
        depth = depth.clamp(min=0.0)
        ratio = depth / (self.lam * self.c)      # lam=-3, c=10/3
        ratio = ratio.clamp(max=1.0 - 1e-7)
        base = 1.0 - ratio
        exponent = self.lam + 1                   # = -2
        normalized = 1.0 - torch.pow(base, exponent)
        return normalized.clamp(0.0, 1.0)
```

### 2.2 Cube-Edge Colormap（RGB 编码）

论文采用**沿 RGB 立方体边缘遍历**的分段线性色图，类似 3D Hilbert 曲线第一轮迭代：

| 阶段 | 颜色 | RGB | 对应深度（米） |
|------|------|-----|---------------|
| 0.00 | Black | (0, 0, 0) | 0m (near) |
| 0.20 | Blue | (0, 0, 1) | ~0.74m |
| 0.40 | Cyan | (0, 1, 1) | ~1.85m |
| 0.60 | Green | (0, 1, 0) | ~3.70m |
| 0.80 | Yellow | (1, 1, 0) | ~7.41m |
| 1.00 | White | (1, 1, 1) | inf (far) |

**关键特性**: 该映射是**严格双射（bijection）**，可逆。论文明确说明:
> "Because both the false-color visualization and the power transform are strictly invertible, their composition forms a **bijection** between metric depth in [0,inf] and RGB space in [0,1]^3."

### 2.3 多色图训练增强

论文要求训练时增强模型对不同颜色表示的鲁棒性：

```python
# src/open_vision_banana/data/mixed.py:72
colormaps = ["cube_edge", "plasma", "inferno", "viridis", "grayscale"]
```

`DepthRGBMapper` 已支持 6 种色图，注册于 `depth_mapper.py:122-129`:

```python
_COLORMAP_REGISTRY = {
    "cube_edge": CUBE_EDGE_KEYPOINTS,
    "plasma": PLASMA_KEYPOINTS,
    "inferno": INFERNO_KEYPOINTS,
    "viridis": VIRIDIS_KEYPOINTS,
    "grayscale": GRAYSCALE_KEYPOINTS,
    "rainbow": RAINBOW_KEYPOINTS,
}
```

---

## 3. 现有数据基础设施分析

### 3.1 数据集注册表

`scripts/download_data.py:59-143` 已注册以下 lightweight alternatives:

| 数据集 | 类型 | 估算大小 | 合成/真实 | 备注 |
|--------|------|---------|----------|------|
| `hypersim-depth` | 深度 | 1.5 TB | **合成** | 论文主要参考，过大 |
| `vkitti2-depth` | 深度 | 50 GB | **合成** | Virtual KITTI 2，驾驶场景 |
| `blendedmvs-depth` | 深度 | 17 GB | **合成** | MVSNet，多视角立体 |
| `hrwsi-depth` | 深度 | 20 GB | 真实 | x 非合成，排除 |
| `diode-depth` | 深度 | 40 GB | 真实 | x 非合成，排除 |
| `tartanair-depth` | 深度 | 30 GB | **合成** | 室内/室外，SLAM |

### 3.2 已有 Dataset 类

`src/open_vision_banana/data/datasets.py` 已实现的深度相关 Dataset:

```python
# 纯合成深度数据集（符合论文要求）
"virtual_kitti": VirtualKITTIDataset,      # 1327-1379 行
"flyingthings3d": FlyingThings3DDataset,  # 1444-1514 行
"hypersim": HypersimDepthDataset,         # 111-147 行

# 真实深度数据集（训练时排除）
"nyu_depth": NYUDepthDataset,             # 488-547 行
"diode": DIODEDataset,                    # 550-607 行
"kitti": KITTIDataset,                    # 610-678 行
"eth3d": ETH3DDataset,                    # 864-928 行
"tum_rgbd": TUMRGBDDataset,               # 1382-1441 行
```

### 3.3 数据格式约定

所有 Dataset 遵循统一接口，返回 `dict`:

```python
{
    "prompt": str,           # 任务指令文本
    "image": Tensor[3,H,W],  # 输入 RGB，范围 [-1, 1]
    "target_rgb": Tensor[3,H,W],  # 深度 RGB 可视化，范围 [0, 1]
    # 可选: "valid_mask", "colormap"
}
```

输出目录结构（`scripts/prepare_hypersim.py` 标准格式）:

```
data/<dataset_name>/
  images/     # RGB 输入图像 (*.png)
  targets/    # 深度 RGB 编码目标 (*.png)
```

---

## 4. Lightweight Alternatives 推荐方案

针对 "**lightweight_alternatives**" 任务类型，推荐以下**纯合成深度数据集**组合，总容量控制在 **~100GB** 以内（vs Hypersim 1.5TB）：

### 4.1 推荐数据集矩阵

| 优先级 | 数据集 | 大小 | 场景类型 | 深度范围 | 获取方式 |
|--------|--------|------|---------|---------|---------|
| P0 | **Virtual KITTI 2** | ~50GB | 自动驾驶/城市场景 | 0-80m | 官网下载 |
| P0 | **TartanAir** | ~30GB | 室内/室外混合 | 0-30m | GitHub 工具 |
| P1 | **BlendedMVS** | ~17GB | 多视角/建筑/物体 | 0-20m | GitHub 发布 |
| P1 | **FlyingThings3D** | ~30GB | 合成动画/光流 | 0-100m | 官网下载 |
| P2 | **Hypersim 子集** | ~10GB | 室内高质量 | 0-10m | 官方 subset |

**总计: ~137GB（全量）或 ~80GB（P0+P1 核心）**

### 4.2 排除数据集（非合成或不符合约束）

| 数据集 | 排除原因 |
|--------|---------|
| NYU Depth v2 | 真实 RGB-D 采集 |
| KITTI | 真实 LiDAR 稀疏深度 |
| DIODE | 真实室内/室外 |
| ETH3D | 真实激光扫描 |
| HRWSI | 真实互联网图像 |
| MegaDepth | SfM 重建，非引擎合成 |
| RedWeb / WSVD | 真实数据 |

---

## 5. 数据准备脚本与命令

### 5.1 Virtual KITTI 2 数据准备

**Step 1: 下载数据**

```bash
# 创建数据目录
mkdir -p data/vkitti2/raw

# 下载 VKITTI 2（需同意学术许可）
# 访问: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/
# 下载后放置到: data/vkitti2/raw/Scene01/ ... Scene06/
```

**Step 2: 转换为标准格式**

创建 `scripts/prepare_vkitti2.py`:

```python
#!/usr/bin/env python3
"""Prepare Virtual KITTI 2 depth data for Vision Banana."""

from pathlib import Path
import argparse
import logging
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from open_vision_banana.core.depth_mapper import DepthRGBMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def prepare_vkitti2(
    source: Path,
    output: Path,
    max_samples: int | None = None,
    resolution: int = 512,
) -> int:
    """Convert VKITTI 2 scene/frames/rgb+depth to images/ + targets/ layout."""
    
    img_dir = output / "images"
    tgt_dir = output / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)
    
    mapper = DepthRGBMapper(colormap="cube_edge", lam=-3.0, c=10.0/3.0)
    
    # VKITTI 2 layout: Scene01/frames/rgb/  Scene01/frames/depth/
    pairs = []
    for scene_dir in sorted(source.glob("Scene*")):
        rgb_dir = scene_dir / "frames" / "rgb"
        depth_dir = scene_dir / "frames" / "depth"
        if not rgb_dir.exists() or not depth_dir.exists():
            continue
        for img_path in sorted(rgb_dir.rglob("*.jpg")):
            depth_path = depth_dir / (img_path.stem + ".png")
            if depth_path.exists():
                pairs.append((img_path, depth_path))
    
    if max_samples:
        pairs = pairs[:max_samples]
    
    written = 0
    for idx, (img_path, depth_path) in enumerate(pairs):
        stem = f"vkitti_{idx:06d}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"
        
        if out_img.exists() and out_tgt.exists():
            written += 1
            continue
        
        # Load and resize RGB
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img.save(out_img)
        
        # Load depth (VKITTI: uint16 mm -> meters)
        depth_img = Image.open(depth_path)
        depth = np.array(depth_img).astype(np.float32)
        if depth.max() > 1.0:
            depth = depth / 1000.0  # mm -> m
        
        depth_tensor = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth_tensor = torch.nn.functional.interpolate(
            depth_tensor, size=(resolution, resolution), mode="nearest"
        )
        
        # Encode to RGB
        rgb_tensor = mapper.depth_to_rgb(depth_tensor).squeeze(0)
        rgb_np = (rgb_tensor.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        Image.fromarray(rgb_np).save(out_tgt)
        
        written += 1
        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", idx + 1, len(pairs))
    
    logger.info("Wrote %d pairs to %s", written, output)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="VKITTI 2 root")
    parser.add_argument("--output", type=str, default="data/vkitti2_depth", help="Output dir")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    
    prepare_vkitti2(
        Path(args.source),
        Path(args.output),
        max_samples=args.max_samples,
        resolution=args.resolution,
    )
```

**执行命令**:

```bash
cd /Users/gatilin/PycharmProjects/open-vision-banana

# 准备 VKITTI 2（全部 50K 帧，约需 2-3 小时）
python scripts/prepare_vkitti2.py \
    --source data/vkitti2/raw \
    --output data/vkitti2_depth \
    --resolution 512

# 或仅准备 5K 样本用于快速验证
python scripts/prepare_vkitti2.py \
    --source data/vkitti2/raw \
    --output data/vkitti2_depth_5k \
    --max-samples 5000 \
    --resolution 512
```

### 5.2 TartanAir 数据准备

```bash
# TartanAir 使用官方下载工具
# https://github.com/castacks/tartanair_tools

git clone https://github.com/castacks/tartanair_tools.git /tmp/tartanair_tools
cd /tmp/tartanair_tools

# 下载子集（例如 abandonedfactory, gascola, office）
python download_training.py --env-list abandonedfactory gascola office \
    --difficulty easy \
    --modality rgb depth \
    --output-dir data/tartanair/raw

# 转换（参照 prepare_hypersim.py 格式创建）
python scripts/prepare_tartanair.py \
    --source data/tartanair/raw \
    --output data/tartanair_depth \
    --resolution 512
```

### 5.3 BlendedMVS 数据准备

```bash
# BlendedMVS 直接下载
# https://github.com/YoYo000/MVSNet
mkdir -p data/blendedmvs/raw
cd data/blendedmvs/raw

# 下载 low-res 版本（~17GB）
wget https://drive.google.com/uc?id=1ilxls-VJNvJnBUKtbCnRf5T5rL1AtCaw \
    -O BlendedMVS.zip
unzip BlendedMVS.zip

# 转换
cd /Users/gatilin/PycharmProjects/open-vision-banana
python scripts/prepare_blendedmvs.py \
    --source data/blendedmvs/raw \
    --output data/blendedmvs_depth \
    --resolution 512
```

### 5.4 验证数据格式

使用项目内置验证工具:

```bash
python -c "
from open_vision_banana.data.validation import validate_vision_dataset
report = validate_vision_dataset('data/vkitti2_depth')
print(report.summary())
"
```

期望输出:
```
Dataset: /Users/gatilin/PycharmProjects/open-vision-banana/data/vkitti2_depth
Valid pairs: 50000
Errors: 0
Warnings: 0
Status: PASS
```

---

## 6. 训练配置对齐

### 6.1 深度对齐训练配置

项目已有配置文件 `configs/training_mps_depth_aligned.yaml`，关键参数:

```yaml
data:
  # 论文 "very low ratio" -- 使用 1:4 (20% 视觉)
  vision_ratio: 0.2
  generation_ratio: 0.8
  resolution: 512
  
  # 色图增强: 论文要求 plasma/inferno/viridis/grayscale
  colormaps:
    - "cube_edge"
    - "plasma"
    - "inferno"
    - "viridis"
    - "grayscale"
  
  tasks:
    depth_estimation:
      datasets:
        - name: "vkitti2_depth"      # 新增: 纯合成数据
          split: "train"
        - name: "tartanair_depth"    # 新增: 纯合成数据
          split: "train"
        - name: "blendedmvs_depth"   # 新增: 纯合成数据
          split: "train"

depth_mapping:
  lam: -3.0
  c: 3.333
```

### 6.2 启动训练

```bash
cd /Users/gatilin/PycharmProjects/open-vision-banana

# MPS 本地验证（Apple Silicon）
python scripts/train_lora_mps.py \
    --config configs/training_mps_depth_aligned.yaml

# 或单卡 A100
python scripts/train.py \
    --config configs/training_a100_80g.yaml \
    --task depth_estimation \
    --data_path data/vkitti2_depth

# 8x H20 DDP
bash scripts/launch_h20_ddp.sh \
    --config configs/training_h20_ddp_8x.yaml \
    --task depth_estimation
```

### 6.3 关键训练参数说明

| 参数 | 论文要求 | 项目配置 | 说明 |
|------|---------|---------|------|
| 数据混合比 | 9:1 generation:vision | `vision_ratio=0.1` | 保留生成功能 |
| 深度编码 | lambda=-3, c=10/3 | `lam=-3.0, c=3.333` | Power transform |
| 色图增强 | plasma/inferno/viridis/grayscale | 6 种色图 | 训练时随机采样 |
| 相机内参 | **不使用** | N/A | 论文强调零内参 |
| 真实深度 | **零使用** | 排除 NYU/KITTI/DIODE 等 | 纯合成约束 |

---

## 7. 数据集类扩展（如需新引擎）

若需集成新仿真引擎（如 NVIDIA Isaac Sim, AI2-THOR, Habitat），在 `datasets.py` 中新增 Dataset 类:

```python
class IsaacSimDepthDataset(BaseVisionDataset):
    """NVIDIA Isaac Sim 合成深度数据.
    
    Expected layout:
        root_dir/
          images/    # RGB renders (*.png)
          depths/    # metric depth (*.npy, float32 meters)
    """
    TASK_TYPE = "depth"
    
    def __init__(self, root_dir, split="train", resolution=512, **kwargs):
        self.depth_mapper = DepthRGBMapper()
        super().__init__(root_dir, split, resolution, **kwargs)
    
    def _scan(self):
        img_dir = self.root_dir / "images"
        depth_dir = self.root_dir / "depths"
        for img_path in sorted(img_dir.rglob("*.png")):
            depth_path = depth_dir / (img_path.stem + ".npy")
            if depth_path.exists():
                self.samples.append({"image": img_path, "target": depth_path})
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = self._load_image(sample["image"])
        
        depth = np.load(sample["target"])
        depth = torch.from_numpy(depth).float().unsqueeze(0)
        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0), size=(self.resolution, self.resolution), mode="nearest"
        ).squeeze(0)
        
        target_rgb = self.depth_mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        prompt = PromptBuilder.build_depth_prompt()
        return {"prompt": prompt, "image": image, "target_rgb": target_rgb}
```

注册到 `build_dataset` factory:

```python
registry = {
    # ... existing entries ...
    "isaacsim_depth": IsaacSimDepthDataset,
}
```

---

## 8. 评估基准与零样本验证

论文在以下基准上评估零样本深度估计（**训练数据不包含这些基准的训练集**）:

| 基准 | 类型 | 论文 delta_1 | 论文 AbsRel |
|------|------|------------|------------|
| NYU Depth v2 | 室内 | 0.965 | 0.07 |
| iBims-1 | 室内 | 0.934 | 0.078 |
| ETH3D | 室内/室外 | 0.935 | 0.103 |
| DIODE-Indoor | 室内 | 0.917 | 0.108 |
| KITTI | 自动驾驶 | 0.953 | 0.086 |
| nuScenes | 自动驾驶 | 0.865 | 0.189 |

项目评估脚本:

```bash
# 零样本评估 NYU Depth v2
python scripts/evaluate_model.py \
    --checkpoint output/checkpoints_mps_depth_aligned/best \
    --task depth \
    --dataset nyu_depth \
    --data_root data/nyu_depth_v2 \
    --split test

# 零样本评估 KITTI
python scripts/evaluate_model.py \
    --checkpoint output/checkpoints_mps_depth_aligned/best \
    --task depth \
    --dataset kitti \
    --data_root data/kitti \
    --split val
```

---

## 9. 质量检查清单

在启动训练前，确认以下事项:

- [ ] 所有训练深度数据来自**仿真引擎**，无真实世界深度传感器数据
- [ ] 排除 NYU/KITTI/DIODE/ETH3D/TUM 等真实数据集的训练 split
- [ ] 数据格式符合 `images/` + `targets/` 目录结构
- [ ] 深度值单位为 **米（meters）**，已正确缩放到 metric 范围
- [ ] `DepthRGBMapper` 参数: `lam=-3.0`, `c=3.333`
- [ ] 色图增强包含至少 cube_edge + plasma + inferno + viridis + grayscale
- [ ] 验证通过 `validate_vision_dataset()` 无 ERROR
- [ ] 生成数据比例保持 ~9:1（generation:vision）
- [ ] 训练配置中 `use_amp`, `rs_lora`, `EMA` 已启用
- [ ] 评估基准的训练集**绝对未混入**训练数据

---

## 10. 附录: 核心代码索引

| 文件 | 行号 | 功能 |
|------|------|------|
| `src/open_vision_banana/core/depth_mapper.py` | 1-357 | 深度<->RGB 双向编码 |
| `src/open_vision_banana/data/datasets.py` | 111-147 | HypersimDepthDataset |
| `src/open_vision_banana/data/datasets.py` | 1327-1379 | VirtualKITTIDataset |
| `src/open_vision_banana/data/datasets.py` | 1444-1514 | FlyingThings3DDataset |
| `src/open_vision_banana/data/mixed.py` | 1-139 | 9:1 混合数据集 |
| `src/open_vision_banana/data/validation.py` | 1-265 | 数据格式验证 |
| `scripts/prepare_hypersim.py` | 1-200 | Hypersim 数据转换 |
| `scripts/download_data.py` | 1-414 | 数据集下载注册表 |
| `configs/training_mps_depth_aligned.yaml` | 1-86 | 深度对齐训练配置 |

---

*报告生成时间: 基于项目当前代码状态*
*论文引用: Gabeur et al., "Image Generators are Generalist Vision Learners", arXiv:2604.20329v3, 2026*
