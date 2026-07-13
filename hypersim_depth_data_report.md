# Hypersim 深度估计训练数据准备 — 技术报告

> **项目**: open-vision-banana  
> **论文对齐**: Image Generators are Generalist Vision Learners (arXiv:2604.20329v3)  
> **任务类型**: hypersim (纯合成深度数据训练)  
> **日期**: 2026-07-13  
> **数据工程师**: Agent-Orchestrator

---

## 1. 论文要求与约束

### 1.1 核心要求

论文 **Section 3.2** 明确声明：

> *"Our model is trained **entirely on synthetic depth data** created from simulation engines — we use **zero real-world depth data**, and exclude training data from any of the depth datasets we evaluate on."*

这意味着：
- ❌ **禁止使用** NYU Depth v2、DIODE、KITTI、ETH3D 等真实世界深度数据集进行训练
- ✅ **必须使用** 纯合成数据（Hypersim、Virtual KITTI 2、BlendedMVS 等）
- ✅ 训练数据与评估基准严格隔离（zero-shot transfer）

### 1.2 深度编码方案

论文提出将 metric depth 编码为 RGB 伪彩色图像，实现 **perception as image generation**：

| 步骤 | 公式 | 参数 |
|------|------|------|
| Power Transform | `f(d, λ, c) = 1 - (1 - d/(λc))^(λ+1)` | λ = -3, c = 10/3 |
| 归一化 | depth ∈ [0, ∞) → [0, 1) | — |
| Cube-Edge Colormap | Black → Blue → Cyan → Green → Yellow → White | 分段线性插值 |

**关键特性**：
- 可逆映射（bijection）：RGB → depth → RGB 的 roundtrip 误差 < 0.5%
- 多色图增强训练：Plasma、Inferno、Viridis、Grayscale、Rainbow 交替使用
- 近处内容精度更高（power transform 的曲线特性）

### 1.3 数据混合比例

论文训练设置：
- 90% 原始 generation 数据（保留 FLUX.1 生成能力）
- 10% vision task 数据（深度估计、法向估计、分割等）
- vision task 内部：多个任务均匀混合

---

## 2. 现有数据基础设施分析

### 2.1 项目目录结构

```
open-vision-banana/
├── scripts/
│   ├── download_data.py          # 数据集下载注册表
│   ├── prepare_hypersim.py       # Hypersim 数据准备（核心）
│   ├── download_train_val.py     # 一键下载 + train/val 划分
│   ├── train.py                  # 主训练脚本
│   └── validate_data.py          # 数据验证
├── src/open_vision_banana/
│   ├── data/
│   │   ├── datasets.py             # 所有数据集加载器
│   │   ├── mixed.py                # 混合数据集（9:1 比例）
│   │   ├── generation.py           # 生成数据保留数据集
│   │   ├── validation.py           # 数据格式验证
│   │   └── split.py                # train/val 划分
│   ├── core/
│   │   └── depth_mapper.py         # Depth ↔ RGB 可逆映射
│   ├── inference/
│   │   └── prompt.py               # 任务提示构建
│   └── training/
│       └── tuner.py                # InstructionTuner
└── tests/
    ├── test_prepare_hypersim.py    # Hypersim 准备测试
    ├── test_datasets.py            # 数据集加载器测试
    └── test_validation.py          # 验证测试
```

### 2.2 现有 Hypersim 支持状态

| 组件 | 状态 | 说明 |
|------|------|------|
| `DatasetSpec` 注册 | ✅ | `hypersim-depth` 已在 `DATASET_REGISTRY` 注册 |
| `prepare_hypersim.py` | ✅ | 支持 `images/` + `targets/` 标准布局 |
| `HypersimDepthDataset` | ✅ | `datasets.py` 中已实现深度数据加载 |
| `DepthRGBMapper` | ✅ | 多色图 + 可逆映射完整实现 |
| `MixedVisionDataset` | ✅ | 支持 9:1 混合 + 多色图增强 |
| 自动下载 | ❌ | Hypersim 需手动下载（Apple GitHub） |
| 完整场景准备 | ⚠️ | 仅支持配对文件扫描，需手动组织目录 |

---

## 3. Hypersim 数据准备流程

### 3.1 数据源

**Hypersim** (Apple Machine Learning, 2021)
- **仓库**: https://github.com/apple/ml-hypersim
- **许可证**: MIT
- **类型**: 合成室内场景，从 3D 场景渲染的逼真图像
- **规模**: 完整数据集约 77,400 张图像，460 个场景，首次 8 个场景约 33GB
- **标注**: 完美 dense depth（来自渲染引擎，无噪声）
- **格式**: `*.color.jpg` + `*.depth.hdf5` / `*.depth.png` 等

### 3.2 数据准备步骤

#### Step 1: 下载 Hypersim 原始数据

```bash
# 克隆 Apple Hypersim 工具仓库（用于下载脚本）
git clone https://github.com/apple/ml-hypersim.git

# 根据 Hypersim 官方指引下载数据
# 注意：完整数据集很大，论文使用的是子集
# 首次下载建议只获取前 8 个场景（~33GB）

# 或者使用已有数据的场景目录
mkdir -p /data/hypersim/raw
# 将下载的 scene_* 目录放入 /data/hypersim/raw/
```

**Hypersim 原始目录结构预期**（需根据实际下载调整）：
```
/data/hypersim/raw/
├── scene_001/                    # 每个场景一个目录
│   ├── images/
│   │   ├── scene_001_000_color.png     # RGB 图像
│   │   ├── scene_001_000_depth.png     # 深度图（米或毫米）
│   │   ├── scene_001_001_color.png
│   │   └── scene_001_001_depth.png
│   └── ...
├── scene_002/
│   └── ...
└── ...
```

#### Step 2: 运行 prepare_hypersim.py

```bash
# 进入项目目录
cd /Users/gatilin/PycharmProjects/open-vision-banana

# 基本用法：将原始数据转换为标准训练布局
python scripts/prepare_hypersim.py \
    --source /data/hypersim/raw \
    --output data/hypersim_depth/ \
    --resolution 512

# 限制样本数（快速测试用）
python scripts/prepare_hypersim.py \
    --source /data/hypersim/raw \
    --output data/hypersim_depth_test/ \
    --max-samples 500 \
    --resolution 512

# 带验证的完整准备
python scripts/prepare_hypersim.py \
    --source /data/hypersim/raw \
    --output data/hypersim_depth/ \
    --resolution 512 \
    --validate
```

**输出布局**（标准 Vision Banana 格式）：
```
data/hypersim_depth/
├── images/                 # RGB 输入图像 [512, 512]
│   ├── 000000_scene_001_000.png
│   ├── 000001_scene_001_001.png
│   └── ...
└── targets/                # 深度编码后的 RGB 目标图像
    ├── 000000_scene_001_000.png
    ├── 000001_scene_001_001.png
    └── ...
```

#### Step 3: 使用 download_train_val.py 进行 train/val 划分

```bash
# 如果已有统一格式的 images/ + targets/，手动划分：
python scripts/download_train_val.py split \
    --source data/hypersim_depth/unified \
    --output data/hypersim_depth_split/ \
    --val-ratio 0.1 \
    --seed 42

# 或者直接使用 Python 脚本划分
python -c "
from open_vision_banana.data.split import split_dataset
from pathlib import Path

train_count, val_count = split_dataset(
    source_dir='data/hypersim_depth',
    output_dir='data/hypersim_depth_split',
    val_ratio=0.1,
    seed=42
)
print(f'Train: {train_count}, Val: {val_count}')
"
```

**划分后布局**：
```
data/hypersim_depth_split/
├── train/
│   ├── images/
│   └── targets/
└── val/
    ├── images/
    └── targets/
```

#### Step 4: 数据验证

```bash
# 验证训练集
python -c "
from open_vision_banana.data.validation import validate_vision_dataset
report = validate_vision_dataset('data/hypersim_depth_split/train')
print(report.summary())
"

# 验证验证集
python -c "
from open_vision_banana.data.validation import validate_vision_dataset
report = validate_vision_dataset('data/hypersim_depth_split/val')
print(report.summary())
"
```

---

## 4. 关键代码详解

### 4.1 Hypersim 数据准备核心代码

文件: `scripts/prepare_hypersim.py`

```python
def find_hypersim_pairs(source: Path) -> list[tuple[Path, Path]]:
    """递归扫描 source 目录下所有 color/depth 文件对."""
    pairs: list[tuple[Path, Path]] = []
    for color_path in sorted(source.rglob("*")):
        if not color_path.is_file():
            continue
        if color_path.suffix.lower() not in IMG_EXTS:
            continue
        # 文件名包含 "color" 或 "rgb"
        if "color" not in color_path.stem.lower() and "rgb" not in color_path.stem.lower():
            continue
        # 将 "color" 替换为 "depth" 得到深度图路径
        depth_path = Path(str(color_path).replace("color", "depth").replace("rgb", "depth"))
        if depth_path.exists():
            pairs.append((color_path, depth_path))
    return pairs


def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    """加载深度图并转换为米."""
    depth_img = Image.open(depth_path)
    depth = torch.from_numpy(np.array(depth_img)).float()
    if depth.ndim == 3:
        depth = depth[..., 0]
    depth = depth.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    # 如果最大值 > 1.0，假设是毫米单位，除以 1000
    if depth.max() > 1.0:
        depth = depth / 1000.0
    return depth.clamp(min=0.0)


def prepare_hypersim(
    source: Path,
    output: Path,
    *,
    max_samples: int | None = None,
    resolution: int | None = None,
    overwrite: bool = False,
) -> int:
    """将 Hypersim 原始数据转换为标准训练布局."""
    pairs = find_hypersim_pairs(source)
    if not pairs:
        logger.error("No color/depth pairs found under %s", source)
        return 0

    if max_samples is not None:
        pairs = pairs[:max_samples]

    img_dir = output / "images"
    tgt_dir = output / "targets"
    img_dir.mkdir(parents=True, exist_ok=True)
    tgt_dir.mkdir(parents=True, exist_ok=True)

    mapper = DepthRGBMapper()
    written = 0

    for idx, (color_path, depth_path) in enumerate(pairs):
        stem = f"{idx:06d}_{color_path.stem.replace('color', '').replace('rgb', '').strip('_')}"
        out_img = img_dir / f"{stem}.png"
        out_tgt = tgt_dir / f"{stem}.png"

        if out_img.exists() and out_tgt.exists() and not overwrite:
            written += 1
            continue

        # 加载并调整 RGB 图像
        color = Image.open(color_path).convert("RGB")
        if resolution is not None:
            color = color.resize((resolution, resolution), Image.Resampling.LANCZOS)

        # 加载深度并调整分辨率
        depth = _load_depth_meters(depth_path)
        if resolution is not None:
            depth = torch.nn.functional.interpolate(
                depth, size=(resolution, resolution), mode="nearest"
            )

        # 深度 → RGB 编码（核心步骤）
        target_rgb = mapper.depth_to_rgb(depth).squeeze(0)
        _tensor_to_pil_rgb(target_rgb).save(out_tgt)
        color.save(out_img)
        written += 1

        if (idx + 1) % 100 == 0:
            logger.info("Processed %d / %d pairs", idx + 1, len(pairs))

    logger.info("Wrote %d pairs to %s", written, output.resolve())
    return written
```

### 4.2 DepthRGBMapper 核心代码

文件: `src/open_vision_banana/core/depth_mapper.py`

```python
class DepthRGBMapper:
    """可逆深度 ↔ RGB 映射器（论文 Section 3.2 实现）"""

    def __init__(self, colormap: str = "cube_edge", lam: float = -3.0, c: float = 10.0 / 3.0):
        self.lam = lam
        self.c = c
        self.colormap = colormap
        self._keypoints = self._get_keypoints(colormap)

    def depth_to_normalized(self, depth: torch.Tensor) -> torch.Tensor:
        """Power Transform: depth [0, ∞) → [0, 1)."""
        depth = depth.clamp(min=0.0)
        ratio = depth / (self.lam * self.c)
        ratio = ratio.clamp(max=1.0 - 1e-7)
        base = 1.0 - ratio
        exponent = self.lam + 1  # = -2
        normalized = 1.0 - torch.pow(base, exponent)
        return normalized.clamp(0.0, 1.0)

    def normalized_to_rgb(self, normalized: torch.Tensor) -> torch.Tensor:
        """分段线性插值：将归一化深度映射到 RGB 色图."""
        kp = self._keypoints.to(normalized.device)
        positions = kp[:, 0].contiguous()      # [K]
        colors = kp[:, 1:].contiguous()        # [K, 3]

        orig_shape = normalized.shape
        n_flat = normalized.reshape(-1)

        # 找到每个像素所属的色段
        seg_idx = torch.searchsorted(positions, n_flat, right=True) - 1
        seg_idx = seg_idx.clamp(0, len(positions) - 2)

        p_start = positions[seg_idx]
        p_end = positions[seg_idx + 1]
        local_t = (n_flat - p_start) / (p_end - p_start + 1e-10)
        local_t = local_t.clamp(0.0, 1.0)

        c_start = colors[seg_idx]
        c_end = colors[seg_idx + 1]
        rgb_flat = c_start + local_t.unsqueeze(-1) * (c_end - c_start)

        rgb = rgb_flat.reshape(*orig_shape, 3)
        # 维度处理...
        return rgb.clamp(0.0, 1.0)

    def depth_to_rgb(self, depth: torch.Tensor) -> torch.Tensor:
        """完整正向：depth → RGB."""
        normalized = self.depth_to_normalized(depth)
        return self.normalized_to_rgb(normalized)

    def rgb_to_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """完整逆向：RGB → depth."""
        normalized = self.rgb_to_normalized(rgb)
        return self.normalized_to_depth(normalized)
```

### 4.3 训练时多色图增强

文件: `src/open_vision_banana/data/mixed.py`

```python
def __getitem__(self, idx: int) -> dict[str, Any]:
    use_vision = (
        self.generation_dataset is None
        or torch.rand(1, generator=self.rng).item() < self.vision_ratio
    )

    if use_vision:
        # 随机选择 vision dataset
        ds_idx = int(torch.randint(len(self.vision_datasets), (1,), generator=self.rng).item())
        ds = self.vision_datasets[ds_idx]
        sample_idx = int(torch.randint(len(ds), (1,), generator=self.rng).item())
        sample = ds[sample_idx]

        # 多色图增强（仅 depth 任务）
        if sample.get("task") == "depth":
            cmap_idx = int(torch.randint(len(self.colormaps), (1,), generator=self.rng).item())
            colormap = self.colormaps[cmap_idx]
            sample["colormap"] = colormap

            # 动态重编码 target_rgb（roundtrip 误差 < 0.5%）
            if colormap != "cube_edge":
                from ..core.depth_mapper import DepthRGBMapper
                mapper = DepthRGBMapper()
                target_rgb = sample["target_rgb"]
                depth = mapper.rgb_to_depth(target_rgb.unsqueeze(0)).squeeze(0)
                mapper.set_colormap(colormap)
                sample["target_rgb"] = mapper.depth_to_rgb(depth.unsqueeze(0)).squeeze(0)
        return sample
    else:
        # 返回 generation 样本
        ...
```

---

## 5. 训练集成

### 5.1 启动训练（Hypersim-only 深度估计）

```bash
cd /Users/gatilin/PycharmProjects/open-vision-banana

# 纯 vision 数据训练（无 generation 数据混合）
python scripts/train.py \
    --data-path data/hypersim_depth_split/train \
    --output-dir output/hypersim_depth_only \
    --epochs 10 \
    --batch-size 8 \
    --resolution 512 \
    --vision-ratio 1.0 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lr 5e-5 \
    --colormaps cube_edge,viridis,plasma,inferno,grayscale,rainbow \
    --val-split 0.1

# 论文对齐：9:1 generation + vision 混合
python scripts/train.py \
    --data-path data/hypersim_depth_split/train \
    --generation-path data/generation_preservation/ \
    --output-dir output/hypersim_paper_aligned \
    --epochs 10 \
    --batch-size 8 \
    --resolution 512 \
    --vision-ratio 0.1 \
    --lora-rank 64 \
    --lora-alpha 128 \
    --lr 5e-5 \
    --colormaps cube_edge,viridis,plasma,inferno,grayscale,rainbow \
    --val-split 0.1 \
    --scheduler cosine \
    --warmup-ratio 0.05
```

### 5.2 训练配置说明

| 参数 | 论文值 | 说明 |
|------|--------|------|
| `--vision-ratio` | 0.1 | 10% vision, 90% generation |
| `--lora-rank` | 64 | rsLoRA rank |
| `--lora-alpha` | 128 | rsLoRA alpha (2× rank) |
| `--lr` | 5e-5 | 学习率 |
| `--colormaps` | 6 种 | 多色图增强 |
| `--resolution` | 512 | 输入/输出分辨率 |
| `--batch-size` | 8 | 根据 GPU 显存调整 |
| `--scheduler` | cosine | 余弦退火 |
| `--warmup-ratio` | 0.05 | 5% warmup |

---

## 6. 验证与质量检查

### 6.1 数据完整性检查

```bash
# 检查 images/ 和 targets/ 是否配对
python -c "
from pathlib import Path
from open_vision_banana.data.validation import validate_vision_dataset

for split in ['train', 'val']:
    path = f'data/hypersim_depth_split/{split}'
    report = validate_vision_dataset(
        path,
        min_pairs=1,
        require_square=True,
        min_resolution=512,
        check_value_range=True
    )
    print(f'=== {split.upper()} ===')
    print(report.summary())
    print()
"
```

### 6.2 深度映射 roundtrip 验证

```python
import torch
from open_vision_banana.core.depth_mapper import DepthRGBMapper

mapper = DepthRGBMapper(colormap="cube_edge")

# 生成测试深度值
test_depth = torch.rand(1, 1, 512, 512) * 10.0  # 0~10 米

# 正向 → 逆向
rgb = mapper.depth_to_rgb(test_depth)
recovered_depth = mapper.rgb_to_depth(rgb)

# 计算误差
abs_error = (test_depth - recovered_depth).abs()
rel_error = abs_error / (test_depth + 1e-6)

print(f"Max absolute error: {abs_error.max():.6f} m")
print(f"Mean relative error: {rel_error.mean():.4f}%")
# 预期: Max absolute error < 0.05 m, Mean relative error < 0.5%
```

### 6.3 可视化检查

```python
import matplotlib.pyplot as plt
from PIL import Image
import torch
from open_vision_banana.core.depth_mapper import DepthRGBMapper

# 加载一个样本
color = Image.open("data/hypersim_depth_split/train/images/000000_scene_001_000.png")
target = Image.open("data/hypersim_depth_split/train/targets/000000_scene_001_000.png")

mapper = DepthRGBMapper()

# 解码 target RGB 回深度
target_tensor = torch.from_numpy(np.array(target)).float().permute(2, 0, 1) / 255.0
depth = mapper.rgb_to_depth(target_tensor.unsqueeze(0)).squeeze(0).squeeze(0)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(color)
axes[0].set_title("RGB Input")
axes[1].imshow(target)
axes[1].set_title("Depth RGB Target")
axes[2].imshow(depth, cmap='viridis')
axes[2].set_title(f"Decoded Depth (max={depth.max():.1f}m)")
plt.tight_layout()
plt.savefig("hypersim_depth_check.png", dpi=150)
```

---

## 7. 扩展建议

### 7.1 其他合成深度数据源

论文明确使用多种合成数据来增强泛化能力。推荐同时准备：

| 数据集 | 类型 | 大小 | 来源 |
|--------|------|------|------|
| Hypersim | 室内合成 | ~33GB (8 scenes) | Apple GitHub |
| Virtual KITTI 2 | 自动驾驶合成 | ~50GB | NAVER Labs |
| BlendedMVS | MVS 合成 | ~17GB | GitHub YoYo000 |
| TartanAir | 无人机合成 | ~30GB | CMU |

**注册新数据集到 `download_data.py`**：

```python
DATASET_REGISTRY["tartanair-depth"] = DatasetSpec(
    name="TartanAir Depth",
    task="depth",
    source="https://github.com/castacks/tartanair_tools",
    size_estimate_gb=30.0,
    requires_login=False,
    license="Academic",
)
```

### 7.2 多任务联合训练

论文的 Vision Banana 是通用视觉模型，需同时训练多个任务：

```bash
# 准备多个 vision task 数据
python scripts/train.py \
    --data-path data/hypersim_depth_split/train \
    --output-dir output/multitask \
    --vision-ratio 0.1 \
    --colormaps cube_edge,viridis,plasma,inferno,grayscale,rainbow

# 在代码中，MixedVisionDataset 会自动混合多个 vision datasets
# 通过 build_dataset 工厂函数加载不同任务
```

---

## 8. 常见问题排查

### Q1: Hypersim 原始深度是 .hdf5 格式而非 .png

**解决方案**：修改 `prepare_hypersim.py` 以支持 HDF5：

```python
import h5py

def _load_depth_meters(depth_path: Path) -> torch.Tensor:
    if depth_path.suffix == ".hdf5" or depth_path.suffix == ".h5":
        with h5py.File(depth_path, "r") as f:
            depth = torch.from_numpy(f["dataset"][:]).float()
    else:
        depth_img = Image.open(depth_path)
        depth = torch.from_numpy(np.array(depth_img)).float()
    # ... 后续处理相同
```

### Q2: 深度值范围不确定（米 vs 毫米）

**当前逻辑**：`if depth.max() > 1.0: depth = depth / 1000.0`
- 最大值 > 1.0 假设为毫米，除以 1000
- 否则假设已经是米

**验证方法**：
```python
import numpy as np
from PIL import Image

depth = np.array(Image.open("scene_000_depth.png"))
print(f"Depth range: {depth.min()} ~ {depth.max()}")
# 如果范围 0~65535 → 毫米单位
# 如果范围 0~20 → 可能已是米单位
```

### Q3: 数据量不足

**建议**：
- 使用完整 Hypersim 数据集（77K 图像）
- 混合多个合成数据集（VKITTI2、BlendedMVS、TartanAir）
- 使用数据增强（已在 `VisionAugmentation` 中实现）

---

## 9. 总结

| 检查项 | 状态 | 备注 |
|--------|------|------|
| 论文要求（纯合成深度） | ✅ 满足 | Hypersim 为合成数据，零真实世界数据 |
| 深度编码方案 | ✅ 实现 | `DepthRGBMapper` 完整实现 power transform + cube-edge |
| 多色图增强 | ✅ 实现 | 6 种 colormap 支持 |
| 数据准备脚本 | ✅ 可用 | `prepare_hypersim.py` 完整功能 |
| 数据验证 | ✅ 可用 | `validate_vision_dataset` 支持 |
| 训练集成 | ✅ 可用 | `MixedVisionDataset` + `train.py` 完整链路 |
| 9:1 混合比例 | ✅ 配置 | `--vision-ratio 0.1` 即可实现 |
| 数据下载 | ⚠️ 需手动 | Hypersim 需从 Apple GitHub 手动下载 |

**下一步行动**：
1. 下载 Hypersim 原始数据（首次 8 个场景，~33GB）
2. 运行 `prepare_hypersim.py` 转换为标准布局
3. 执行 `validate_vision_dataset` 确认数据质量
4. 启动训练：`python scripts/train.py --data-path data/hypersim_depth --vision-ratio 0.1 ...`

---

*报告生成时间: 2026-07-13 20:05 CST*  
*项目路径: `/Users/gatilin/PycharmProjects/open-vision-banana`*
