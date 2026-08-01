# v3 数据集下载与可用性状态报告

**生成时间**: 2026-07-31 17:50
**项目**: open-vision-banana
**数据集根目录**: `/Users/gatilin/MyWork/datasets`
**磁盘空间**: 4.1TB 可用 (已用 3.3TB / 7.3TB)

---

## 1. 总体可用性

| 指标 | Tier 0 修复前 | Tier 0 修复后 | 本次修复后 |
|------|-------------|-------------|-----------|
| 可用样本 | 16,248 (27.9%) | 20,058 (34.4%) | 21,079 (36.1%) |
| 缺失样本 | 42,083 | 38,273 | 37,252 |
| 净增 | - | +3,810 | +4,831 |

## 2. 按任务族可用性

| 任务 | 可用 | 总量 | 可用率 | 变化 | 优先级 |
|------|------|------|--------|------|--------|
| depth | 5,221 | 9,000 | 58.0% | +2.4% | MEDIUM |
| normal | 5,000 | 5,500 | 90.9% | - | OK |
| segmentation | 3,717 | 23,831 | 15.6% | +1.2% | HIGH |
| keypoint | 2,141 | 13,500 | 15.9% | +0.8% | HIGH |
| multiview | 0 | 1,000 | 0.0% | - | HIGH |
| other | 5,000 | 5,500 | 90.9% | - | OK |
| **TOTAL** | **21,079** | **58,331** | **36.1%** | **+1.7%** | |

## 3. 已完成操作

### 3.1 解压完成

| 数据集 | 文件 | 解压状态 | 提取文件数 |
|--------|------|---------|-----------|
| Objects365 | 14个 patch tar.gz (13GB) | 已解压 | ~70,000+ (patch0=23K) |
| DOTA | images.zip (1.9GB) | 已解压(7z) | 4,249/4,250 (1个损坏) |
| VIPSeg | VIPSeg.zip (968MB) | 损坏，已删除重新下载中 | 0 |

### 3.2 路径映射修复

| 数据集 | JSONL路径 | 实际路径 | 修复方式 | 效果 |
|--------|----------|---------|---------|------|
| BDD100K | `BDD100K/100k/train/xxx.jpg` | `BDD100K/data/xxx.jpg` | 符号链接 981 个 | 无效（文件名不匹配） |
| VisDrone | `VisDrone2019-DET-train/images/xxx.jpg` | `VisDrone2019-DET/data/xxx.jpg` | 符号链接 862 个 | 部分有效 (+97) |
| DOTA | `train/images/Pxxxx.png` | `extracted/images/train/Pxxxx.png` | 符号链接 4,249 个 | 无效（版本不匹配） |
| Objects365 | `patchN/xxx.jpg` | `extracted/patchN/xxx.jpg` | 符号链接 14 patches | 部分有效 (+221) |
| LoveDA | `Train/Rural/images_png/xx.png` | `images_png/xx.png` + `urban_rural_train_images/` | 符号链接 +209 | 有效 (+169) |

### 3.3 后台下载任务

| 数据集 | 状态 | 下载文件数 | 备注 |
|--------|------|-----------|------|
| FSC147 | 失败 (401) | 1,808 | HF API 需要认证 |
| VIPSeg | 下载中 | 0 | 重新下载中 |
| Seg数据集(TAS500等) | 子代理执行中 | - | 后台运行 |

## 4. 未解决问题

### 4.1 数据集版本不匹配

| 数据集 | 问题 | 影响 | 建议 |
|--------|------|------|------|
| BDD100K | JSONL引用的文件名与HF下载的子集不匹配 | 1,000个keypoint样本 | 需下载BDD100K完整100k训练集 |
| DOTA | JSONL引用原图名(P1113)，下载的是裁剪子图(P0000_1_1) | 1,000个keypoint样本 | 需下载DOTA v1.0原图集 |
| Objects365 | 仅下载patch0-13，JSONL引用patch0-44 | 2,779个depth样本 | 需继续下载patch14-44 |
| FSC147 | HF API返回401 Unauthorized | 736+720个keypoint样本 | 需HF token认证 |
| DeepFashion2 | 下载返回0字节 | 1,000个keypoint样本 | 需换源 |
| Flickr30k | nlphuji/flickr30k仅含3527张子集 | 889个segmentation样本 | 需找完整31783张源 |
| WildRGB-D | 目录不存在，未启动下载 | 500个multiview样本 | 需从wildrgb.github.io下载 |
| VIPSeg | 首次下载zip不完整(968MB) | 1,000个segmentation样本 | 重新下载中 |
| Hypersim | 仅8个场景，JSONL引用ai_001~ai_044 | 4,000个segmentation样本 | 需下载303个完整场景 |
| ScanNet++ | 需签署ToS许可 | 1,500个depth+normal样本 | 手动获取 |
| ScanNet v2 | 需签署ToS许可 | 500个multiview样本 | 手动获取 |

### 4.2 缺口最大的数据集（按缺失样本数排序）

| 排名 | 数据集 | 任务 | 缺失 | 原因 |
|------|--------|------|------|------|
| 1 | hypersim_genseg | segmentation | 3,000 | 仅8个场景 |
| 2 | object365_depth | depth | 2,779 | 仅14/45 patches |
| 3 | cityscapes_gcg | segmentation | 2,000 | 需登录下载 |
| 4 | cityscapes_genseg | segmentation | 2,000 | 需登录下载 |
| 5 | BDD100K | keypoint | 1,000 | 版本不匹配 |
| 6 | DeepFashion2 | keypoint | 1,000 | 下载失败 |
| 7 | dota_point | keypoint | 1,000 | 版本不匹配 |
| 8 | 51world_genseg | segmentation | 1,000 | 未下载 |
| 9 | IDD-1_genseg | segmentation | 1,000 | 未下载 |
| 10 | vipseg_gcg | segmentation | 1,000 | zip损坏 |

## 5. 下一步行动计划

### Phase 1: 快速修复（预计2小时）
1. 继续下载 Objects365 patch14-44（~57GB，后台进行）
2. 下载 Flickr30k 完整图像（换用 HuggingFaceM4/flickr30k）
3. 下载 WildRGB-D（从 GitHub releases）

### Phase 2: 中等修复（需人工介入）
1. 获取 HF token 后重新下载 FSC147
2. 下载 BDD100K 完整 100k 训练集（需正确repo）
3. 下载 DOTA v1.0 原图集（非裁剪版）
4. 下载 DeepFashion2（换源）

### Phase 3: 大型数据集（需大量带宽和时间）
1. 下载 Hypersim 303个场景（~450GB）
2. 下载 VIPSeg 完整数据集（~10GB）

### Phase 4: 需许可数据集（手动获取）
1. ScanNet++ (https://kaldir-vc.github.io/scannetpp/)
2. Cityscapes (https://www.cityscapes-dataset.com/login/)
3. nuImages (https://www.nuscenes.org/download)
4. ScanNet v2 (http://www.scan-net.org/)

## 6. 可用性验证脚本

验证脚本位置: `scripts/check_v3_availability.py`

```bash
# 重新运行验证
python3 scripts/check_v3_availability.py

# 报告输出
# docs/v3_availability_report.md  - Markdown格式
# docs/v3_availability_report.json - JSON格式
```

## 7. 关键经验教训

1. **HF API认证**: 部分数据集repo需要token认证（如FSC147），需提前配置 `HF_TOKEN`
2. **版本匹配**: JSONL中的文件名与HF下载的文件名可能不匹配（BDD100K、DOTA），需先检查JSONL路径格式再下载对应版本
3. **zip完整性**: 大文件下载可能不完整，需验证zip central directory后再使用
4. **路径前缀**: JSONL路径和实际存储路径经常不一致，需建立符号链接映射
5. **Objects365规模**: 单个patch就有23K+文件，45个patches总计可能超过100万文件，需选择性下载JSONL引用的子集
