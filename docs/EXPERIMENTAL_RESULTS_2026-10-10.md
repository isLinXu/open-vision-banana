# Open-Vision-Banana 实验结果（2026-10-10）

论文：[Image Generators are Generalist Vision Learners](https://arxiv.org/abs/2604.20329)（arXiv 2604.20329v3）。  
评测机：`gpu4090-1`，部署树 `/comp_robot/home/xu.lin/workspaces/ovb-paper-eval-20261009/`。  
指标 JSON：`output/eval/`。

**这不是 Table 1 复现。** 论文数字只出现在「论文」列，用来对照。本仓库用开源基座 + 公开 LoRA + 项目自定闸门，没有 Nano Banana Pro，也没有论文未公开的训练数据。

官方 Cityscapes val（500 对 frankfurt/lindau/munster，`gtFine`）已从 ModelScope `OpenDataLab/CityScapes` 装到评测机。论文口径的 Cityscapes 数字是 SANA CS-19 15k `checkpoint-8000_best` 的 mIoU **0.266**（n=500；3000 步 specialist 是 0.177）。旧的 ADE-150 LoRA 交叉色板分数 0.024 / 0.015 只作对照。ReasonSeg / SA-Co/Gold / GenAI-Bench / ImgEdit 评测集已在评测机；nuScenes trainval 仍在下载。协议缺口见第 6 节。

---

## 1. 结论

1. **2D。** RefCOCOg UMD val：SANA `ovb-fix_seg_plus_gcg` cIoU **0.143**，FLUX `ovb-flux3_ctrl_seg` cIoU **0.098**，论文 0.738。Cityscapes val（论文口径）：SANA `sana_1600m_cityscapes19_15k` `checkpoint-8000_best` mIoU **0.266**（3000 步 specialist 是 0.177），论文 0.699。ADE-150 LoRA 交叉色板对照仍是 SANA 0.024 / FLUX 0.015，不再当论文数字。ReasonSeg、SA-Co/Gold 没有可引用分数。
2. **3D 深度。** NYU、iBims-1、ETH3D、DIODE-Indoor、KITTI 都按官方划分、原生 GT 分辨率、像素加权 δ1 / AbsRel 算过了。SANA 相对灰度、SANA Hypersim 度量、FLUX.1-dev 相对灰度、FLUX.1 Kontext Hypersim EMA 四条线的度量闸门全部失败。Kontext NYU δ1 **0.025** / AbsRel **3.027**，比 SANA / FLUX.1-dev 灰度还差。
3. **3D 法向。** 只有 DIODE-Indoor。SANA `ovb-fix_ctrl_normal` 均值 **24.96°**，过了项目法向闸门，不是论文的 13.818°。FLUX.1-dev 法向均值 **91.09°**，接近随机朝向。
4. **生成保留。** 论文是人对人胜率。这边没有 GenAI-Bench / ImgEdit 人类评估，不能报 53.5% / 47.8%。
5. **统一模型。** 下面每条分数都来自任务 LoRA，不是同一个共享权重。不能当成论文「一个生成器切任务」的证据。

四数据集（NYU + ETH3D + DIODE-Indoor + KITTI）算术平均 δ1，仅作对照，论文没有写清它怎么平均：

| 模型 | 4-set δ1 | 5-set δ1（再加 iBims） | 论文 |
|------|--------:|----------------------:|-----:|
| SANA `fix_ctrl_depth` 灰度近处更亮 | 0.149 | 0.140 | 4-set 0.929 / 6-set 0.882 |
| SANA Hypersim cube-edge 度量 | 0.163 | 0.166 | 同上 |
| FLUX.1-dev `flux3_ctrl_depth` 灰度近处更亮 | 0.138 | 0.146 | 同上 |

---

## 2. 评测口径

入口：`scripts/evaluate_paper_benchmarks.py`。分组：`--benchmark 2d|3d|generation|all`。

| 项目 | 约定 | 是不是论文原文 |
|------|------|----------------|
| 分辨率 | `--native-resolution`，在未缩放 GT 上计分 | 论文没写死；社区常用 |
| 深度聚合 | 像素加权为标题数字；`*_per_image` 和 `*_aligned` 是诊断 | 论文没写用哪种平均 |
| `delta_1.25_aligned` | 逐图中位数尺度对齐后再算 δ1 | **不是**论文指标 |
| NYU | Eigen crop + 10 m | 社区约定 |
| KITTI | Garg crop + 80 m | 社区约定 |
| ETH3D | 80 m，无 Eigen/Garg crop | 社区约定 |
| nuScenes | 80 m，无 crop | 论文没披露 crop；适配器已接，数据不在 |
| DIODE | 只要 indoor | 论文是 DIODE-Indoor |
| 法向 | 相机系，`R=(1-x)/2, G=(1+y)/2, B=(1+z)/2` | 对齐论文编码 |
| RefCOCOg | UMD val，每句一个样本，polygon/RLE，不用框 | 对齐论文划分 |
| ReasonSeg | LISA `jpg+json`，指标 gIoU | 论文还用 Gemini 2.5 Pro 改写查询 |
| SA-Co/Gold | `cgF1 = 100 × pmF1 × IL_MCC` | 论文还用 Gemini 3.1 Flash-Lite 做 NP 过滤 |
| Cityscapes | 19 类 trainId，官方 `gtFine` | 对齐；**禁止** `cityscapes8` |
| 生成 | 论文是人类 pairwise | CLIP/FID 只许标 proxy |

深度闸门（项目自定，不是论文阈值）：度量 δ1 至少比最佳非 oracle 平凡基线高 0.10，且 AbsRel 不差于该基线。结构诊断看 `delta_1.25_aligned` 对逐图中位数 oracle。

法向闸门（项目自定）：均值角误差至少比 `facing_camera` 低 5°，并低于逐图均值 oracle。

推理：SANA 20 步、fp32、guidance 1.0。FLUX.1-dev 20 步、4-bit transformer + 8-bit T5、训练匹配 guidance 50.0。Kontext 28 步、4-bit、cube-edge。相对深度 LoRA 用 `--depth-colormap grayscale --invert-depth`。度量 cube-edge 不用 invert。

FLUX.1-dev 第一次评测修过三处，重跑分数才有效：蒸馏 transformer 要 `guidance` tensor；sequential offload 时 VAE 不能在 meta 上 decode；4-bit/8-bit 文本编码器不能再 `.to(float32)`。

---

## 3. 模型与权重

| 简称 | 基座 | 权重 | 训练目标 | 评测用法 |
|------|------|------|----------|----------|
| SANA gray depth | SANA-1600M | [khazic/ovb-fix_ctrl_depth](https://huggingface.co/khazic/ovb-fix_ctrl_depth) EMA step 30000 | 近处更亮灰度 | 5 个深度集 |
| SANA metric depth | SANA-1600M | 本地 `sana_1600m_hypersim_metric_depth` EMA step 3000 | Hypersim 米制 cube-edge | 5 个深度集 |
| SANA multitask depth | SANA-1600M | `ovb-fix_best3_s42` EMA step 30000 | 多任务，不是深度 specialist | NYU / DIODE 对照，不进主表 |
| SANA normal | SANA-1600M | [khazic/ovb-fix_ctrl_normal](https://huggingface.co/khazic/ovb-fix_ctrl_normal) EMA step 30000 | 法向 RGB | DIODE-Indoor |
| SANA referring | SANA-1600M | [khazic/ovb-fix_seg_plus_gcg](https://huggingface.co/khazic/ovb-fix_seg_plus_gcg) adapter step 30000，无 EMA | 分割 + GCG | RefCOCOg UMD val |
| SANA semantic（交叉色板对照） | SANA-1600M | [khazic/ovb-fix_ctrl_seg](https://huggingface.co/khazic/ovb-fix_ctrl_seg) adapter step 30000，无 EMA | SNVC ADE-150 | 官方 Cityscapes val，**不是**论文口径 |
| SANA semantic CS-19 | SANA-1600M | 本地 `sana_1600m_cityscapes19_15k` `checkpoint-8000_best`（Best val_loss 0.0266） | 官方 CS-19 色图 | 官方 Cityscapes val（论文口径） |
| FLUX semantic（交叉色板对照） | FLUX.1-dev | [khazic/ovb-flux3_ctrl_seg_nominsnr_15k](https://huggingface.co/khazic/ovb-flux3_ctrl_seg_nominsnr_15k) step 15000 | SNVC ADE-150 | 官方 Cityscapes val，**不是**论文口径 |
| FLUX depth | FLUX.1-dev | [khazic/ovb-flux3_ctrl_depth_nominsnr_15k](https://huggingface.co/khazic/ovb-flux3_ctrl_depth_nominsnr_15k) step 15000 | 近处更亮灰度 | 5 个深度集 |
| FLUX normal | FLUX.1-dev | [khazic/ovb-flux3_ctrl_normal_nominsnr_15k](https://huggingface.co/khazic/ovb-flux3_ctrl_normal_nominsnr_15k) step 15000 | 法向 RGB | DIODE-Indoor |
| FLUX Kontext depth | FLUX.1-Kontext-dev | 本地 Hypersim 闸门 run EMA step 3000（`flow_loss=0.1526`） | 米制 cube-edge | NYU Eigen + 10 m，闸门 FAIL |

Cityscapes 数据来自 ModelScope `OpenDataLab/CityScapes` 的 `raw/gtFine_trainvaltest.zip` 和 `raw/leftImg8bit_trainvaltest.zip`，解 train+val（2975 + 500）。没有用 `cityscapes8`。

---

## 4. 按论文能力轴

### 4.1 2D understanding

论文：Cityscapes val mIoU、RefCOCOg UMD val cIoU、ReasonSeg val gIoU、SA-Co/Gold cgF1。

| 基准 | 论文 | 本仓库 | 状态 |
|------|-----:|--------|------|
| RefCOCOg UMD val cIoU | 0.738 | 0.143 / 0.098 | 已测（SANA / FLUX） |
| RefCOCOg UMD val gIoU | — | 0.158 / 0.092 | 已测（论文主报 cIoU） |
| RefCOCOg Precision@0.5 | — | 0.089 / 0.0006 | 已测 |
| ReasonSeg val gIoU | 0.793 | — | 适配器已接；评测机无数据。论文分数含 Gemini 2.5 Pro 改写 |
| Cityscapes val mIoU | 0.699 | **0.266** | 已测（SANA CS-19 15k `checkpoint-8000_best`；3000 步 0.177；交叉色板对照 0.024 / 0.015） |
| SA-Co/Gold cgF1 | 47.5 | — | 适配器已接 query JSONL 和 cgF1。无 Gold 数据。论文分数含 Gemini 存在性过滤 |

RefCOCOg 细节：UMD val，每句一个样本，原生 mask，n=4896。

- SANA-1600M `ovb-fix_seg_plus_gcg` 非 EMA adapter，20 步，fp32。cIoU **0.143**，gIoU 0.158，Precision@0.5 0.089。产物：`refcocog_umd_val_fix_seg_plus_gcg_native.json`。
- FLUX.1-dev `ovb-flux3_ctrl_seg_nominsnr_15k` `checkpoint-15000_final`，20 步，4-bit，bf16。cIoU **0.098**，gIoU 0.092，Precision@0.5 0.0006，Recall 0.500。产物：`refcocog_umd_val_flux3_ctrl_seg_native.json`。这个 LoRA 是语义分割权重，不是指代 specialist；CLIP 77 token 截断也会吃掉长表达式。

Cityscapes 细节：官方 val 500 对（frankfurt 267 / lindau 59 / munster 174），`labelIds → trainIds`，19 类全局混淆矩阵，512² 最近邻缩放（`--native-resolution` 对 semantic 不生效）。这是色图最近邻解码后的类 ID mIoU，不是可视化观感。Prompt 是官方 19 色 legend（`SEMANTIC_SEGMENTATION_PROMPT`）。

- **论文口径（当前最好）。** SANA-1600M `sana_1600m_cityscapes19_15k` `checkpoint-8000_best/lora_adapter`，从 3000 步权重只加载 LoRA 再训到 15000，Best val_loss **0.0266**（step 8000；之后 9k–15k val 未刷新）。20 步，fp32，guidance 1.0。mIoU **0.266**，pixel acc **0.731**，n=500。大面积类：road 0.812，sky 0.819，vegetation 0.711，building 0.585，car 0.493，terrain 0.339，sidewalk 0.333，person 0.288。细/稀有类仍弱：motorcycle 0.003，truck 0.012，wall 0.023，traffic_light 0.031，rider 0.035。产物：`cityscapes_val_cityscapes19_15k.json`。
- **3000 步对照。** 同一色板 `checkpoint-3000_best` mIoU **0.177**，pixel acc 0.572。产物：`cityscapes_val_cityscapes19.json`。
- **交叉色板对照，不是论文口径。** SANA `ovb-fix_ctrl_seg` mIoU **0.024**。FLUX `ovb-flux3_ctrl_seg_nominsnr_15k` mIoU **0.015**。ADE-150 训、CS-19 评。产物：`cityscapes_val_fix_ctrl_seg.json`，`cityscapes_val_flux3_ctrl_seg.json`。

0.266 比交叉色板 0.024 高一个数量级，比 3000 步 0.177 再涨约 50%。仍远低于论文 0.699；主导类已可辨，边界和稀有类还差。评测会拒绝 ADE-150 prompt 的 LoRA（除非 `--allow-semantic-palette-mismatch`）。

### 4.2 3D understanding — 度量深度

论文 6 集：NYU、iBims-1、ETH3D、DIODE-Indoor、KITTI、nuScenes。这里有前 5 集。

#### 4.2.1 SANA `ovb-fix_ctrl_depth` EMA，灰度，近处更亮，20 步

| 数据集 | n | δ1 | AbsRel | aligned δ1 | 闸门 |
|--------|--:|---:|-------:|-----------:|:----:|
| NYU Eigen + 10 m | 654 | 0.066 | 1.990 | 0.427 | FAIL |
| DIODE-Indoor | 325 | 0.083 | 550.1 | 0.322 | FAIL |
| KITTI Garg + 80 m | 1000 | 0.264 | 0.436 | 0.471 | FAIL |
| ETH3D public DSLR + 80 m | 454 | 0.186 | 1.765 | 0.418 | FAIL |
| iBims-1 | 100 | 0.101 | 229.5 | 0.345 | FAIL |

NYU / DIODE 闸门 JSON：`nyu_ctrl_depth_grayflip_gate.json`、`diode_ctrl_depth_grayflip_gate.json`。KITTI / ETH3D / iBims 的灰度闸门未另存文件；用同一规则对照第 5 节平凡基线，同样失败。

#### 4.2.2 SANA Hypersim 度量 cube-edge EMA，step 3000，20 步，不 invert

| 数据集 | n | δ1 | AbsRel | aligned δ1 | 闸门 |
|--------|--:|---:|-------:|-----------:|:----:|
| NYU Eigen + 10 m | 654 | 0.124 | 1.251 | 0.323 | FAIL |
| DIODE-Indoor | 325 | 0.184 | 18.62 | 0.494 | FAIL |
| KITTI Garg + 80 m | 1000 | 0.051 | 0.773 | 0.241 | FAIL |
| ETH3D public DSLR + 80 m | 454 | 0.292 | 0.543 | 0.486 | FAIL |
| iBims-1 | 100 | 0.179 | 23.02 | 0.393 | FAIL |

ETH3D 这条 δ1 0.292 略高于常数@3 m 的 0.272，但仍低于门槛 0.372，AbsRel 0.543 也差于该基线的 0.453。

#### 4.2.3 FLUX.1-dev `ovb-flux3_ctrl_depth_nominsnr_15k`，4-bit，灰度近处更亮，20 步

| 数据集 | n | δ1 | AbsRel | aligned δ1 | 闸门 |
|--------|--:|---:|-------:|-----------:|:----:|
| NYU Eigen + 10 m | 654 | 0.155 | 1.472 | 0.202 | FAIL |
| DIODE-Indoor | 325 | 0.140 | 2433 | 0.219 | FAIL |
| KITTI Garg + 80 m | 1000 | 0.095 | 0.778 | 0.178 | FAIL |
| ETH3D public DSLR + 80 m | 454 | 0.163 | 1.398 | 0.206 | FAIL |
| iBims-1 | 100 | 0.179 | 2096 | 0.206 | FAIL |

DIODE / iBims 的 AbsRel 到了 10³ 量级：cube-edge / 灰度解出的远处接近饱和，少数像素主导误差。aligned δ1 也只有 0.18–0.22，说明中位数对齐之后空间结构仍然弱。

#### 4.2.4 FLUX.1 Kontext Hypersim EMA — 已测，闸门 FAIL

- 训练：3000 / 3000，`flow_loss=0.1526`，Best val_loss 仍为 `inf`（验证没选出 checkpoint）。评测用 `checkpoint-3000_final/ema_adapter`。
- 第一次 NYU：部署树路径不存在，`nyu_flux_kontext_metric_cube_native.json` 只有失败 preflight，**没有分数**。
- 有效评测：`--nyu-root .../open-vision-banana/data/benchmarks/nyu_v2/official_test`，28 步，4-bit，bf16，cube-edge，不 invert，prompt `Generate a metric depth visualization of this image.` n=654。

| 数据集 | n | δ1 | AbsRel | aligned δ1 | 闸门 |
|--------|--:|---:|-------:|-----------:|:----:|
| NYU Eigen + 10 m | 654 | 0.025 | 3.027 | 0.431 | FAIL |

闸门：`nyu_flux_kontext_ema_gate.json`。度量闸门失败（模型 δ1 0.025，要求 ≥ `constant@3m` 0.352 + 0.10 = 0.452；AbsRel 3.027 也差于该基线 0.357）。结构闸门失败（aligned δ1 0.431 < 逐图中位数 oracle 0.519）。比 SANA / FLUX.1-dev 灰度 NYU 都差。产物：`nyu_flux_kontext_ema_metric_cube_native.json`。

#### 4.2.5 对照：`ovb-fix_best3_s42`（不进主表）

多任务 EMA，不是深度 specialist。NYU 灰度 δ1 0.084 / AbsRel 1.815；DIODE 灰度 δ1 0.086 / AbsRel 11911。默认色图不 invert 更差（NYU δ1 0.015）。

### 4.3 3D understanding — 表面法向

论文 Table 7：室内均值 15.549° / 中位 9.300°；NYU 17.778 / 8.876；DIODE-Indoor 13.818 / 11.556；ScanNet 15.052 / 7.468；VKITTI 29.063 / 10.699。四集合计 18.928°。

| 模型 | 数据集 | n | mean° | median° | 11.25° | 22.5° | 30° | 闸门 |
|------|--------|--:|------:|--------:|-------:|------:|----:|:----:|
| SANA `fix_ctrl_normal` EMA | DIODE-Indoor | 325 | 24.955 | 17.008 | 0.313 | 0.626 | 0.727 | PASS |
| FLUX `flux3_ctrl_normal` | DIODE-Indoor | 325 | 91.092 | 91.507 | 0.008 | 0.036 | 0.065 | FAIL |

SANA 相对 `facing_camera` 43.62° 低了约 18.7°，也低于逐图均值 oracle 35.37°。这只说明它比平凡法向基线有结构，不说明复现了论文。

NYU / ScanNet / VKITTI 官方法向图不在评测树。不从深度估算法向。

### 4.4 Visual generation

| 基准 | 论文 | 本仓库 |
|------|------|--------|
| GenAI-Bench T2I 人类胜率 | 53.5% vs NBP 46.5% | 未跑。适配器只读 prompt。CLIP/FID 不是这个指标 |
| ImgEdit 人类胜率 | 47.8% vs NBP 52.2% | 未跑。适配器读 image+instruction |

没有人类 pairwise，就没有生成保留结论。

---

## 5. 平凡基线（原生分辨率）

闸门用的是「最佳非 oracle」。oracle 只作上界。

### 5.1 深度

| 数据集 | 最佳非 oracle | δ1 | AbsRel | 逐图中位数 oracle δ1 |
|--------|---------------|---:|-------:|---------------------:|
| NYU | constant@3 m | 0.352 | 0.517 | 0.519 |
| DIODE-Indoor | constant@2 m | 0.329 | 0.458 | 0.692 |
| KITTI | constant@10 m | 0.340 | 0.356 | 0.337 |
| ETH3D | constant@3 m | 0.272 | 0.453 | 0.562 |
| iBims-1 | constant@3 m | 0.318 | 0.395 | 0.570 |

NYU 闸门 JSON 里 `reference_absrel=0.357` 对应 constant@2 m 的 AbsRel，`reference_delta1=0.352` 对应 constant@3 m。下表闸门结论仍以写出的 JSON 为准。

度量闸门门槛（δ1 ≥ 基线 + 0.10）：NYU 0.452、DIODE 0.429、KITTI 0.440、ETH3D 0.372、iBims 0.418。三条模型线全部低于门槛。

### 5.2 法向（DIODE-Indoor）

| 基线 | mean° | median° |
|------|------:|--------:|
| facing_camera | 43.621 | 41.555 |
| axis_up | 90.402 | 89.119 |
| oracle_dataset_mean | 43.323 | 41.418 |
| oracle_image_mean | 35.371 | 29.786 |

SANA 24.96° 过闸。FLUX 91.09° 和 `axis_up` 一个量级。

---

## 6. 未测或阻塞

| 能力 | 基准 | 原因 |
|------|------|------|
| 2D | ReasonSeg val | 评测机已有 200 张 val（`paper_benchmarks/reasonseg`，jpg + mask png + `annotations.json`）。缺 Gemini 改写，不能对论文 0.793 |
| 2D | SA-Co/Gold | 图像和三份官方标注已下。`queries.jsonl` 用 annotator A，不是三标注 oracle。缺 Gemini NP 过滤，不能对论文 47.5 |
| 3D | nuScenes | 官方 trainval（含 val）正在下，约 290 GB。下完还要把 LiDAR 投到 `CAM_FRONT` 的 `depth/*.npy`，适配器才能算 80 m δ1 |
| 3D | NYU / ScanNet / VKITTI 法向 | 无社区/官方法向图 |
| 3D | Kontext NYU | 已测：δ1 0.025 / AbsRel 3.027，闸门 FAIL |
| 生成 | GenAI-Bench / ImgEdit | 评测集已在评测机。论文指标是人类 pairwise，不是 CLIP/FID |
| 统一模型 | 共享 checkpoint 五任务 | 没有同一权重的 2D+3D+生成结果 |

远程有 RefCOCOg（`/comp_robot/home/xu.lin/data/paper_benchmarks/refcocog`）。FLUX 分割 LoRA 的 UMD val 已测，cIoU 0.098。

---

## 7. 和论文数字的差距（不要外推成「还差多少训练」）

| 指标 | 论文 | 本仓库最好的已测值 | 差距 |
|------|------|-------------------|------|
| 4-set 深度 δ1 | 0.929 | 0.163（SANA metric） | 0.77 |
| NYU δ1 | 0.948 | 0.155（FLUX gray） | 0.79 |
| DIODE-Indoor 法向 mean° | 13.818 | 24.955（SANA） | +11.1° |
| RefCOCOg cIoU | 0.738 | 0.143（SANA） | 0.595 |
| Cityscapes mIoU | 0.699 | 0.266（SANA CS-19 15k） | 0.433 |
| SA-Co cgF1 | 47.5 | 无 | — |
| ReasonSeg gIoU | 0.793 | 无 | — |
| GenAI-Bench 胜率 | 53.5% | 无 | — |

基座、数据、混合比、更新范围都和论文不同。这些差不能换算成「再训多少步」。

---

## 8. 产物

评测 JSON 在 `ovb-paper-eval-20261009/output/eval/`：

- 深度主结果：`{nyu,diode_indoor,kitti,eth3d,ibims}_{fix_ctrl_depth_ema_grayflip,hypersim_metric_cube,flux3_ctrl_depth_grayflip}_native.json`
- 法向：`diode_indoor_fix_ctrl_normal_ema_native.json`，`diode_indoor_flux3_ctrl_normal_native.json`
- 指代：`refcocog_umd_val_fix_seg_plus_gcg_native.json`，`refcocog_umd_val_flux3_ctrl_seg_native.json`
- 语义（论文口径）：`cityscapes_val_cityscapes19_15k.json`（当前最好）；3000 步对照 `cityscapes_val_cityscapes19.json`
- 语义（交叉色板对照）：`cityscapes_val_fix_ctrl_seg.json`，`cityscapes_val_flux3_ctrl_seg.json`
- 闸门 / 平凡基线：`*_gate.json`，`*_trivial_native.json`
- Kontext NYU（有效）：`nyu_flux_kontext_ema_metric_cube_native.json`，闸门 `nyu_flux_kontext_ema_gate.json`
- Kontext 第一次无效：`nyu_flux_kontext_metric_cube_native.json`（路径错误，无分数）

代码：`scripts/evaluate_paper_benchmarks.py`；2D/3D/生成分组见 `src/open_vision_banana/evaluation/paper_capabilities.py`。ReasonSeg / GenAI-Bench / ImgEdit 适配器在 `src/open_vision_banana/data/paper_eval_datasets.py`。nuScenes 深度现在返回 `gt_depth`。

---

## 9. 补记

- [x] Kontext NYU：n=654，δ1 0.025，AbsRel 3.027，aligned δ1 0.431，闸门 FAIL
- [x] CS-19 15k 训完：Best val_loss **0.0266**（step 8000）。官方 val mIoU **0.266** / pixel acc 0.731，JSON：`cityscapes_val_cityscapes19_15k.json`
- [x] FLUX `ovb-flux3_ctrl_seg_nominsnr_15k` RefCOCOg UMD val：n=4896，cIoU **0.098**，gIoU 0.092，Precision@0.5 0.0006
- [ ] 不把 CLIP/FID、cityscapes8、或失败 preflight 写进主表
