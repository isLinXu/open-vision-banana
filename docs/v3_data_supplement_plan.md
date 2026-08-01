# v3 训练集数据扩充方案

**日期**: 2026-07-30
**项目**: open-vision-banana
**当前版本**: v3_optimized (56,839 样本 / 12 任务)
**目标**: 系统性下载缺失源图像，扩充训练数据

---

## 1. 现状分析

### 1.1 SNVC50M JSONL 源图像可用性

对 57 个已下载的 JSONL 标注文件（58,331 个样本）进行逐文件验证：

| 指标 | Tier 0 修复前 | Tier 0 修复后 |
|------|--------------|--------------|
| 可用样本 | 16,248 (27.9%) | 20,058 (34.4%) |
| 缺失样本 | 42,083 | 38,273 |
| 路径修复回收 | - | +3,810 |

### 1.2 按任务族的缺失分布（Tier 0 修复后）

| 任务 | v3 当前样本 | JSONL 总量 | 已可用 | 缺失 | 可用率 | 优先级 |
|------|-----------|-----------|--------|------|--------|--------|
| depth | 2,296 | 9,000 | 5,000 | 4,000 | 55.6% | HIGH |
| normal | 1,796 | 5,500 | 5,000 | 500 | 90.9% | MEDIUM |
| segmentation | 7,483 | 25,231 | 3,430 | 21,801 | 13.6% | HIGH |
| keypoint | 9,407 | 11,600 | 628 | 10,972 | 5.4% | HIGH |
| object_detection | 9,433 | 1,000 | 1,000 | 0 | 100% | OK |
| referring | 1,690 | 5,000 | 5,000 | 0 | 100% | OK |
| multiview | 965 | 1,000 | 0 | 1,000 | 0% | HIGH |

> 注：`specular_removal`、`structure`、`generation`、`instance`、`depth_edge` 不使用 SNVC50M JSONL 数据源，故标注为 N/A。

### 1.3 Tier 0 路径修复详情

以下数据集的图像**已存在**于本地，仅路径前缀不匹配，通过修改 JSONL 中的路径字段修复：

| JSONL 名称 | 修复前可用 | 修复后可用 | 修复方式 |
|------------|-----------|-----------|---------|
| grocery_bbox | 0/500 | 500/500 | 添加 `GroceryStore/` 前缀 |
| grocery_visual | 0/500 | 500/500 | 添加 `GroceryStore/` 前缀 |
| paco_point | 0/500 | 500/500 | 添加 `coco2017/` 前缀 |
| loveda_gcg | 0/1500 | 1109/1500 | 扁平化到 `LoveDA/images_png/` |
| loveda_genseg | 0/1500 | 1201/1500 | 扁平化到 `LoveDA/images_png/` |

**LoveDA 部分缺失原因**: 当前 LoveDA 目录只有 2,031 张扁平 PNG，但 JSONL 引用了 3,000 个不同文件名。需重新下载完整 LoveDA Train+Val 数据集（约 1.4GB）。

---

## 2. 下载方案

### 2.1 分层策略

| 层级 | 描述 | 预计回收样本 | 预计下载量 |
|------|------|------------|-----------|
| Tier 0 | 路径映射修复（已完成） | +3,810 | 0 |
| Tier 1A | 小型公开数据集 (<3GB each) | ~8,000 | ~15GB |
| Tier 1B | 中型公开数据集 (3-15GB each) | ~12,000 | ~45GB |
| Tier 1C | 大型公开数据集 (>15GB) | ~10,000 | ~70GB |
| Tier 2 | 需许可/注册数据集 | ~6,000 | 需手动获取 |
| 合计 | | ~39,810 | ~130GB+ |

### 2.2 Tier 1A: 小型公开数据集（优先下载）

这些数据集每个小于 3GB，可快速下载并立即提升样本量：

| 数据集 | 任务 | 大小 | 来源 | 许可证 |
|--------|------|------|------|--------|
| TAS500 | segmentation | 0.5GB | `HF: dataset-ninja/tas500` | MIT |
| KITTI semseg | segmentation | 0.5GB | `captain-whu.github.io/DOTA` | CC-BY 3.0 |
| TrashCan | segmentation | 0.5GB | `github.com/QAnimal/TrashCan` | MIT |
| TTPLA | segmentation | 0.3GB | `github.com/he-y/TTPLA` | MIT |
| UDD5 | segmentation | 0.2GB | `github.com/IPIS-MSRA/Underwater-Drone-Dataset` | MIT |
| UDD6 | segmentation | 0.2GB | 同上 | MIT |
| StreetHazards | segmentation | 1GB | `qanimal.org/streethazards` | MIT |
| ZeroWaste-f | segmentation | 1GB | `github.com/dbash/zerowaste` | MIT |
| LoveDA (full) | segmentation | 1.4GB | `HF: yangwang825/LoveDA` | MIT |
| Flickr30k | segmentation | 1.3GB | `HF: nlphuji/flickr30k` | CC-BY 2.0 |
| PID-Ray | segmentation | 1GB | `github.com/RsTU-Lab/PIDRay` | MIT |
| APTv2 | keypoint | 2GB | 学术网站 | Academic |
| FSC147 (full) | object_detection | 3GB | `HF: nlphuji/fsc147` | MIT |

**Tier 1A 预期效果**: 修复后 ~8,000 个新样本可用

### 2.3 Tier 1B: 中型公开数据集

| 数据集 | 任务 | 大小 | 来源 | 许可证 |
|--------|------|------|------|--------|
| BDD100K (100k) | keypoint | 7GB | `HF: BDD100K/bdd100k` | BSD-2-Clause |
| VisDrone2019 | keypoint | 5GB | `HF: Shubhendu404/VisDrone2019-DET-train` | MIT |
| DeepFashion2 | keypoint | 10GB | `HF: deepfashion2/deepfashion2` | MIT |
| VIPSeg | segmentation | 10GB | `github.com/VIPSeg/VIPSeg-Dataset` | CC-BY 4.0 |
| IDD (v1+v2) | segmentation | 6GB | `idd.insaan.iiit.ac.in` | MIT |
| WildRGB-D | multiview | 30GB | `wildrgb.github.io` | CC-BY 4.0 |

### 2.4 Tier 1C: 大型公开数据集

| 数据集 | 任务 | 大小 | 来源 | 许可证 |
|--------|------|------|------|--------|
| Objects365 v2 | depth | ~70GB | `HF: OpenGVLab/InternImage` | Apache 2.0 |
| DOTA v1.0 | keypoint | 15GB | `captain-whu.github.io/DOTA` | Academic |
| FAIR1M | keypoint | 20GB | `gaofen-challenge.com` | Academic |
| V3Det | keypoint | 50GB | `HF: V3Det/V3Det` | Apache 2.0 |
| Hypersim (303 scenes) | segmentation | ~450GB | `HF: org-llnl/hypersim` | CC-BY-SA 4.0 |

### 2.5 Tier 2: 需许可/注册数据集

| 数据集 | 任务 | 大小 | 获取方式 | 许可证 |
|--------|------|------|---------|--------|
| ScanNet++ | depth + normal | 1.5TB | `kaldir-vc.github.io/scannetpp` 签署 ToS | 非商业研究 |
| Cityscapes (full) | segmentation | 11GB | `cityscapes-dataset.com` 登录 | 非商业研究 |
| nuImages | keypoint | 50GB | `nuscenes.org` 注册 | 非商业研究 |
| ScanNet v2 | multiview | 1.2TB | `scan-net.org` 签署 ToS | 非商业研究 |

---

## 3. 推荐执行顺序

### Phase 1: 快速见效（Tier 0 + Tier 1A, ~1 天）

```bash
# 已完成: Tier 0 路径修复 (+3,810 samples)
# 下一步: 下载小型数据集
python scripts/download_v3_supplement.py --tier 1 --task segmentation
python scripts/download_v3_supplement.py --tier 1 --task keypoint
python scripts/download_v3_supplement.py --tier 1 --task object_detection
```

**预期效果**: 可用样本从 20,058 提升到 ~28,000

### Phase 2: 中等扩充（Tier 1B, ~2-3 天）

```bash
# 下载中型数据集 (BDD100K, VisDrone, DeepFashion2 等)
# 注意: 需要稳定的网络连接和足够的磁盘空间
```

**预期效果**: 可用样本提升到 ~40,000

### Phase 3: 大型数据集（Tier 1C, 按需）

```bash
# 按需下载 Objects365, Hypersim scenes 等
# 建议: 只下载 JSONL 引用的子集，而非全量
```

### Phase 4: 许可数据集（Tier 2, 手动）

需要用户手动在以下网站注册并下载:
1. **ScanNet++**: https://kaldir-vc.github.io/scannetpp/
2. **Cityscapes**: https://www.cityscapes-dataset.com/login/
3. **nuImages**: https://www.nuscenes.org/download
4. **ScanNet v2**: http://www.scan-net.org/

---

## 4. 扩充后均衡性评估

### 4.1 预期样本量（Phase 1+2 完成后）

| 任务 | v3 当前 | Phase 1 后 | Phase 2 后 | 理想范围 |
|------|---------|-----------|-----------|---------|
| depth | 2,296 | 2,296 | 5,296 (+Objects365) | 5,000+ |
| normal | 1,796 | 1,796 | 2,296 (+ScanNet++ if licensed) | 3,000+ |
| segmentation | 7,483 | 15,483 | 25,000+ | 10,000+ |
| keypoint | 9,407 | 11,407 | 20,000+ | 10,000+ |
| object_detection | 9,433 | 10,433 | 10,433 | 10,000+ |
| referring | 1,690 | 1,690 | 1,690 | 2,000+ |
| multiview | 965 | 965 | 1,465 (+WildRGB-D) | 2,000+ |
| specular_removal | 564 | 564 | 564 | 1,000+ |
| structure | 2,598 | 2,598 | 2,598 | 3,000+ |
| generation | 5,981 | 5,981 | 5,981 | 6,000+ |
| instance | 5,581 | 5,581 | 5,581 | 6,000+ |
| depth_edge | 9,045 | 9,045 | 9,045 | 9,000+ |

### 4.2 均衡性改善

| 指标 | v3 当前 | Phase 1+2 后 |
|------|---------|-------------|
| 最大/最小比 | 16.7x (9433/564) | ~17x (25000/564) |
| sqrt加权后最大/最小比 | 4.1x | ~6.5x |
| <1000 样本的任务 | 1 (specular_removal) | 1 (specular_removal) |
| <2000 样本的任务 | 4 | 2-3 |

> **注意**: specular_removal 和 multiview 仍然是短板。specular_removal 不在 SNVC50M 覆盖范围内，需要从其他来源（如 lllab 数据集）补充。multiview 依赖 ScanNet v2（需许可）和 WildRGB-D。

---

## 5. 数据集整合方案

### 5.1 新数据并入 v3 的流程

```
1. 下载原始图像 → /Users/gatilin/MyWork/datasets/{dataset_name}/
2. 验证 JSONL 路径可访问性 (运行验证脚本)
3. 运行 build_v3_dataset.py 重新构建:
   python scripts/build_v3_dataset.py \
     --dataset-root /Users/gatilin/MyWork/datasets \
     --output-root /Users/gatilin/PycharmProjects/open-vision-banana/data/vb_snvc_train_v4
4. 运行 selfcheck 验证:
   python data/vb_snvc_train_v3/selfcheck_v3.py
5. 重建 MD5 索引和 train/val split
```

### 5.2 版本管理

- 当前版本: `v3_optimized` → 保持不变
- 扩充后版本: `v4_expanded` (新建目录，不覆盖 v3)
- 建议保留 v3 作为回退基线

### 5.3 关键注意事项

1. **跨任务 MD5 防泄漏**: 新数据必须参与全局 MD5 索引重建，确保同一图像不跨 split 出现
2. **色板统一**: 新 segmentation 数据必须映射到 ADE20K 110 色统一色板
3. **Prompt 多样化**: 新样本需要应用 5 模板变体生成 prompt（参见 `diversify_prompts.py`）
4. **Task weights**: 扩充后需重新评估 sqrt 加权采样比例

---

## 6. 可执行脚本

下载脚本已生成: `scripts/download_v3_supplement.py`

```bash
# 查看完整计划（不执行）
python scripts/download_v3_supplement.py --tier all --dry-run

# 执行路径修复（已完成）
python scripts/download_v3_supplement.py --tier 0

# 下载小型数据集
python scripts/download_v3_supplement.py --tier 1 --task segmentation

# 下载全部公开数据集
python scripts/download_v3_supplement.py --tier 1

# 按任务过滤
python scripts/download_v3_supplement.py --tier 1 --task keypoint
```
