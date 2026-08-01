# Open-Vision-Banana 深度分析报告 v1.1.0

> 分析日期：2026-07-26 · 基于 v0.4.2 源码 + 当日训练实测数据
> 前置基线：`OPEN_VISION_BANANA_DEEP_ANALYSIS_v100.md`、`DEEP_ANALYSIS_INDEPENDENT_20260726.md`（今晨独立核验版）
> 本版增量：**纳入真实训练实验数据**（SSD-1B source-conditioned depth、specular removal、overfit gate、32×100 多样本验证、decoded-x0/空间 Adapter、标量排序、冻结特征判别头，以及 VAE-only/SSD held-out 消融），将「复现完成度」评估从静态推断升级为实证跟踪。

---

## 一、项目总览

### 1.1 定位与愿景

基于 Vision Banana 论文（*"Image Generators are Generalist Vision Learners"*）的开源实现：将视觉任务输出参数化为 RGB 图像，用图像生成模型（diffusion）以 img2img「重绘」出任务结果，再解码回结构化输出。**不改架构，仅换 prompt**，一个模型统一深度/法线/分割等 10 类任务。

与论文的关键差异：**论文基座是 Google 闭源 Nano Banana Pro；本仓库用开放权重模型替代**（FLUX.1-schnell 默认，另有 SSD-1B、SDXL、Z-Image、Qwen-Image、SANA 等 23 个基座家族注册）。

### 1.2 项目规模

| 维度 | 数值 |
|------|------|
| 版本 | 0.4.2（SemVer + Keep-a-Changelog，9 个版本迭代） |
| 源码 | 69 个 Python 文件，约 20.5K 行（`src/open_vision_banana/`） |
| 测试 | 50 个测试文件；CHANGELOG 记录 467+ 通过 |
| 脚本 | 92 个 CLI 脚本（训练/评估/下载/可视化/发布） |
| 配置 | 23 个 YAML（MPS/H20/A100/DDP/FSDP/DeepSpeed 全覆盖） |
| 基础设施 | Docker（CPU+GPU）、3 个 GitHub Actions workflow、pre-commit、ruff+mypy |

### 1.3 与相关工作的位置

| 对比项 | 判别式专用模型（Depth Anything / SAM） | 论文 NBP | 本仓库 |
|--------|--------------------------------------|----------|--------|
| 输出空间 | 任务特定头 | RGB 统一 | RGB 统一 ✓ |
| 基座可用性 | 开放 | **闭源** | 开放（23 家族） |
| 已验证精度 | SOTA | SOTA（论文数字） | **未实证** |
| 工程完整度 | 高 | N/A | 高（训练/评估/导出/服务全链路） |

---

## 二、架构深度解析

### 2.1 分层架构（依赖方向单向向下）

```
scripts/ (92 CLI)
   │
   ▼
serving/ ── FastAPI（FastAPI 惰性导入）
export/  ── ONNX / TorchScript
training/ ── InstructionTuner + 5 后端（Native/DDP/FSDP/DeepSpeed/Accelerate）
inference/ ── VisionBanana pipeline、generator（img2img）、prompt、decoder
evaluation/ ── 10 任务指标 + benchmark + baselines
core/ ── 零外部依赖的可逆 RGB 映射算法（CPU 可独立使用）
model_registry.py ── 23 基座家族的统一抽象（pipeline 类 / text encoder 数 /
                     added_cond_kwargs / VAE 参数 / LoRA target / 显存预算）
```

**设计亮点：**
- `core/` 与重依赖完全解耦——算法层可以在无 GPU、无 diffusers 的环境独立运行和测试。
- `__init__.py` 全量 `__getattr__` 惰性导入，核心包 import 不触发 torch/diffusers 加载。
- `model_registry.py` 把 23 个架构差异巨大的扩散家族（UNet vs DiT、1~3 个 text encoder、added_cond_kwargs 有无、guidance/attention_mask/position_ids/pooled_projections 等 8 种 forward 签名变体）收敛到单一 `ModelConfig` dataclass——这是本仓库相对一般「论文复现脚本」最显著的工程增量。

### 2.2 核心算法层（与论文对齐，实测可逆）

- **深度**：幂变换 `f(d)=1-(1+d/10)^(-2)`（论文 Eq.1，λ=-3, c=10/3）+ cube-edge 色图 + 6 种增强色图；独立实测 roundtrip 相对误差 0.0000%。
- **法线**：右手系线性映射，roundtrip cosine sim = 1.000004。
- **实例分割**：5 阶段聚类（Appendix A，τ=14, θ=2e-4, γ=5.0），Union-Find 加速（256² 实测 137×，验证论文 ~100× 声明）；v0.4.0 起 S4/S5 阶段向量化，病态输入 430× 提速。
- **自适应深度**：v0.3.x 新增 adaptive sampling（06-18 benchmark：512² 快 1.45×，roundtrip 误差 0.34%）。

### 2.3 训练流水线

- rsLoRA（α/√r scaling），LoRA target 默认 `to_q/k/v, to_out.0, proj_in/out, ff.net.0/2`（v0.3.x 修复了仅 `to_k/to_v` 的旧缺陷）。
- 5 后端统一 `TrainerInterface`；EMA、early stopping、latent cache、lr finder、梯度检查点、checkpoint 自动剪枝完备。
- img2img 范式已正确实现（`FluxImg2ImgPipeline`），source-conditioned 8-channel 输入（ip2p 风格 conv_in 扩展）已在 SSD-1B 上跑通 checkpoint 往返。

### 2.4 推理与部署

- 5 个优化 preset（balanced/speed/memory/quality/cpu）、ONNX 导出（mapper 纯张量 + full pipeline）、FastAPI serving、Docker 双镜像。

---

## 三、当日实证进展（本报告核心增量）

> 今晨独立报告的结论是「复现完成度 D+：无权重、未评估、范式未实证」。**当天下午的实际训练数据显示项目已从「脚手架」进入「实证攻坚」阶段**，以下是基于 `output/` 实测数据的核验：

### 3.1 正在进行的训练（截至 17:23，仍在运行）

**SSD-1B specular_removal LoRA**（`output/ssd_specular_focused/`）：
- 配置：MPS，bf16，LoRA rank=64/alpha=128，rsLoRA，512px，1168 样本，14 epochs / 2044 steps
- 进度：Step 1730/2044（85%），loss 2.7094 → 0.7220，cosine LR 正常衰减，EMA + checkpoint 剪枝正常工作
- 吞吐：约 **74 s/step**（512px, bs=1, grad_accum=8）——全程约 42 小时，MPS 单卡吞吐是明确的效率瓶颈

### 3.2 Source-conditioned depth 攻坚（`output/quick_validate_ssd_v3/`）

这是回答「范式是否成立」的主线实验，采用 ip2p 风格 8-channel source conditioning，取得**首个正向学习信号**：

| 阶段 | 结果 | 判定 |
|------|------|------|
| 单样本 gate（4-step 推理调度采样） | Pearson r=0.8154（几何结构已学到），RGB MAE=0.1337 | 几何 ✓ / 色彩 ✗ |
| 逐通道 affine 诊断 | 同一预测 MAE 降至 0.0409 | 残差主要是系统性色彩/动态范围标定，非结构缺失 |
| 力矩标定 + viridis 曲线投影 | MAE=0.0563，r=0.9351（16 样本标定集） | 正向信号 |
| 全量 2162 训练目标重算标定 | MAE=0.0883 | 16 样本前缀不具代表性（已修正为全量统计） |
| 生产 smoke（1 step，30 官方验证样本） | r=-0.05 | **闭环验证 ✓，泛化未证明**（符合 smoke 预期） |

### 3.3 数据审计的重要发现（P0 级）

`docs/plans/2026-07-26-ssd-depth-quality-stage2-design.md` 记录的审计结果：

1. **v3 训练标签实为 viridis 色图，而非论文的 cube-edge 幂变换编码**——32 图审计：viridis 最近曲线误差 0.0075 vs cube-edge 0.3631。manifest 与 prompt 声称的编码方式与数据不符（已修正为「检测色图 + 匹配 prompt」）。
2. **v3 无原始 metric depth 与逐图 scale 元数据**——该数据集只支持 relative-depth 评估，**metric depth 不可从此数据重建**。论文的 AbsRel/δ1 复现必须换用 Hypersim/NYU 等带真值的数据。
3. **train/val split 泄漏风险已堵**：数据修复后的当前 manifest 声明 2182 train / 115 val（总数仍为 2297；旧报告中的 2162 / 135 已过时），旧 CLI 加载全部配对；已修正为默认只加载声明的 train。

### 3.4 诚实的负结果记录

- **Overfit gate 未通过**（`overfit_gate_depth/overfit_gate_results.json`）：16 样本 300 步过拟合测试，SSIM=0.058（阈值 0.8）、PSNR=6.75、L1=0.377，`gate_passed=false`。该结果被完整记录而非掩盖——这是工程诚信的体现，也说明 epsilon-prediction 在少步推理调度下的色彩精度问题尚未根治。
- Pixel L1 辅助损失在 MPS 上「技术可行但不够稳定」（1 步改善 0.1337→0.1324，8 步退回 0.1338），未设为默认。

### 3.5 下一阶段多样本验证（23:42 更新）

采用 `segmind/SSD-1B`、128px、LoRA rank/alpha=8/8、4 个部署时刻、
epsilon + x0（权重 1.0），对当前 split 做了新的分层止损实验：

| 实验 | Raw RGB MAE | 相对位置 MAE | Pearson | Spearman | 结论 |
|------|-------------|--------------|---------|----------|------|
| 32 train，step 20，16 val | 0.2635 | 0.3611 | -0.0567 | — | 起点 |
| 32 train，step 60，16 val | 0.2453 | 0.3449 | -0.0652 | — | 颜色改善，几何未改善 |
| 32 train，step 100，16 val | 0.2407 | 0.3399 | -0.0593 | -0.0843 | 触发 300-step 止损 |
| 同 step 100，16 train | 0.2445 | 0.3417 | 0.0037 | 0.0015 | 不是普通过拟合；训练集也未学到排序 |
| 8 train × 200 steps，8 train | 0.2306 | 0.3491 | -0.0873 | -0.0830 | 增加单样本曝光仍无几何记忆 |
| 8 train × 100 steps，LR=5e-4，8 train | 0.2139 | 0.3273 | 0.0134 | 0.0141 | 更强更新只继续改善颜色 |

历史 production step-100 checkpoint 也用同一 16 条验证样本重新核验：Raw RGB
MAE=0.2123，但 Pearson=0.0147、Spearman=0.0009。此前 30 条样本上的
Pearson=0.1073 受样本选择与高方差影响，不能视为稳定泛化证据。

8 通道输入卷积的 source 半区并非未训练：其权重范数相对原始输入半区从
低 LR step-100 的 0.67% 增至高 LR 探针的 3.77%。因此当前证据指向：
**source 信号确实进入网络，但仅在首层拼接、rank-8 LoRA 与现有扩散损失下，
优化首先学到目标颜色/范围统计，尚未学到稳定的空间深度映射。** 下一阶段不应
盲目追加到 300 步，应先比较结构感知 decoded-x0 辅助损失与多尺度条件注入
（T2I-Adapter/ControlNet）的小样本 geometry gate。

### 3.6 Geometry gate 对照结果（次日 00:28 更新）

在前述止损结论后，已完成两种结构增强路径的实现与 matched 小样本对照：

1. **结构感知 decoded-x0**：冻结 VAE 解码，加入多尺度 Charbonnier 值域损失与
   水平/垂直梯度匹配；
2. **T2I-Adapter 多尺度注入**：`full_adapter_xl`，约 82.8M 参数，4 个零初始化
   1×1 投影通过 UNet 原生 `down_intrablock_additional_residuals` 注入。

三组统一使用 4 train / 4 val、128px、60 optimizer steps、rank/alpha=8/8、
LR=5e-4、4 inference steps、seed 42、no EMA。以下为同一评估器得到的均值：

| 组别 | Split | Raw RGB MAE | Raw 位置 MAE | Raw Pearson | Raw Spearman | Cal. Pearson | Cal. Spearman |
|------|-------|-------------|--------------|-------------|--------------|--------------|---------------|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 | 0.0108 | 0.0293 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 | 0.0528 | -0.0415 |
| decoded-x0 结构损失 | train | 0.2222 | 0.3215 | -0.0277 | -0.0276 | -0.0232 | -0.0213 |
| decoded-x0 结构损失 | val | 0.2171 | 0.2800 | 0.0816 | -0.0346 | 0.1036 | -0.0231 |
| 多尺度 Adapter | train | 0.2158 | 0.3042 | 0.0004 | 0.0163 | 0.0208 | -0.0022 |
| 多尺度 Adapter | val | 0.2199 | 0.3146 | -0.0040 | -0.1114 | 0.0287 | -0.0600 |

**两组均未通过 geometry gate。** decoded-x0 组的验证 Pearson 有小幅改善，且
校准后验证 RGB/位置 MAE 改善，但 train Pearson/Spearman 同时退化、验证
Spearman 仍为负；在每组仅 4 个样本且相关系数标准差约 0.25–0.38 的条件下，
不能将其视为几何学习证据。Adapter 组的验证 Pearson/Spearman 均退化，尤其
Raw Spearman 从 -0.0411 降至 -0.1114。

Adapter 失败不是接线问题：60 步后四个零卷积权重已全部非零，范数约为
0.84/1.66/3.09/3.13，fresh-process 严格加载也已通过。其 checkpoint（含优化器）
约 547MB，而另外两组约 58MB，未换来验证几何收益。因此当前证据进一步指向：
**RGB 局部结构损失和更深空间容量都不是充分条件，训练目标仍未直接约束相对深度
排序。** 不应立即放大数据、步数或升级完整 ControlNet；下一小门应将 decoded RGB
可微映射到 viridis 标量曲线位置，直接优化标量梯度与 ordinal/rank consistency，
通过 train 排序记忆后再与 Adapter 组合。

### 3.7 标量位置与排序目标验证（01:01 更新）

为直接对齐 relative-depth 评估口径，新增可微 soft colormap projection：将
decoded-x0 RGB 投影到 viridis 曲线标量位置，并联合优化标量 Charbonnier、
水平/垂直标量梯度及确定性长距离 ordinal loss。该实现不使用随机 pair sampling，
因此不会改变 matched baseline 的训练随机数序列。

同一 4 train / 4 val、128px、60-step 协议测试了总权重 0.2 和 1.0：

| 组别 | Split | Raw RGB MAE | Raw 位置 MAE | Raw Pearson | Raw Spearman | Cal. Pearson | Cal. Spearman |
|------|-------|-------------|--------------|-------------|--------------|--------------|---------------|
| Baseline | train | 0.2182 | 0.3254 | -0.0076 | 0.0247 | 0.0108 | 0.0293 |
| Baseline | val | 0.2124 | 0.2741 | 0.0705 | -0.0411 | 0.0528 | -0.0415 |
| 标量几何 0.2 | train | 0.2133 | 0.3118 | 0.0260 | 0.0533 | 0.0700 | 0.0751 |
| 标量几何 0.2 | val | 0.2111 | 0.2845 | 0.0478 | -0.0656 | 0.0354 | -0.0627 |
| 标量几何 1.0 | train | 0.2123 | 0.3065 | 0.0272 | 0.0185 | 0.0741 | 0.0681 |
| 标量几何 1.0 | val | 0.2122 | 0.2884 | 0.0520 | -0.0548 | 0.0580 | -0.0259 |

0.2 组的 train Raw Pearson/Spearman 同向小幅改善，但增量远小于约
0.20–0.35 的组内标准差，而且在 validation 上反向。之所以只追加一个 1.0 权重
探针，是因为 0.2 在同 seed 的单步总 loss 上仅比 baseline 高约 0.022；结果显示
更强权重并未增强 train Raw Spearman，validation 仍退化，排除了简单欠权重解释。

**Stage 5 未通过。** 结构损失、空间 Adapter 和直接标量排序三条路径均无法让
当前 SSD-1B/rank-8 LoRA 扩散路径稳定记忆四条训练图的深度排序。继续调 loss、
加步数或堆 ControlNet 的信息价值已经很低。下一 bounded gate 应在冻结 SSD/VAE
特征上接一个小型判别式标量深度头，首先只要求 4-image overfit，以区分「基座
没有可用几何表征」与「迭代 RGB diffusion 无法有效读出表征」。若判别头也不能
记忆，应停止 SSD-1B 深度主线，改用预训练单目深度 backbone 或 metric-depth 数据。

### 3.8 冻结 SSD/VAE 特征判别头（01:29 更新）

为把「基座没有几何表征」与「diffusion RGB 读出失败」拆开，新增完全冻结的
SSD-1B 特征探针：VAE latent 加 UNet 三个 down block、mid block 和 final 输出，
经独立 1×1 投影、FPN 融合及三层上采样，直接输出单通道相对深度。基座只做一次
确定性 `t=0` 前向并缓存特征，训练阶段仅更新 509,745 参数的解码头。

真实 SSD-1B/MPS 四图 gate 使用 128px、seed 42、LR=3e-3、最多 2000 步，
在第 75 步提前通过：

| Step | 标量 MAE | Pearson | Spearman |
|-----:|---------:|--------:|---------:|
| 1 | 0.1649 | 0.5239 | 0.5399 |
| 25 | 0.0740 | 0.9438 | 0.9398 |
| 50 | 0.0575 | 0.9619 | 0.9608 |
| 75 | **0.0330** | **0.9736** | **0.9740** |

四张图逐图也全部通过 `MAE<=0.05, Pearson/Spearman>=0.95`；最低 Pearson
为 0.9566，最大 MAE 为 0.0421。全流程含基座加载、四图特征缓存、训练与产物
写入仅 12.0 秒，checkpoint 已按六层 feature contract 严格重载。

**Stage 6 通过，不能停止 SSD-1B 深度主线。** 这项结果证明冻结 SSD/VAE 表征
中存在可被轻量头快速读出的几何信息，因而 Stage 3–5 的失败应更准确归因于
迭代 RGB diffusion 的读出/优化，而非「基座完全没有几何」。但四图记忆不代表
泛化：下一阶段必须做 split-clean 的 VAE-only 与 VAE+SSD matched 对照，只有 SSD
中间特征在 held-out Pearson/Spearman 上优于 VAE-only，才值得继续扩展判别路径；
该结果不支持恢复 ControlNet、Adapter 或 diffusion loss 调参。

### 3.9 VAE-only vs VAE+SSD held-out 消融（02:10 更新）

Stage 6 的四图记忆同时使用了 VAE latent 与 SSD 中间特征，不能证明收益来自
SSD。为隔离贡献，Stage 7 使用同一个六输入、509,745 参数 head，并在 projection、
GroupNorm 和激活之后施加 mask：VAE-only 为 `[1,0,0,0,0,0]`，VAE+SSD 为全开。
两组严格共享初始化 checksum、batch schedule、32 train / 16 official val、128px、
seed 42、batch 8、LR=1e-3 及固定 300-step endpoint。

| Arm | Split | 标量 MAE | Pearson | Spearman |
|-----|-------|---------:|--------:|---------:|
| VAE-only | train | 0.1256 | 0.8694 | 0.8569 |
| VAE-only | val | 0.2591 | 0.1425 | 0.1629 |
| VAE+SSD | train | 0.0670 | 0.9602 | 0.9535 |
| VAE+SSD | val | 0.2402 | **0.3663** | **0.3682** |

VAE+SSD 相对 VAE-only 的 validation 增量为 Pearson `+0.2238`、Spearman
`+0.2053`，同时 MAE 改善 `0.0190`、RMSE 改善 `0.0145`；两项排序增量均超过
预设 `+0.05` 门槛的 4 倍。全流程含一次 48 图特征缓存和两个 arm 训练共 36.3 秒。

**Stage 7 明确通过。** 这是首次 split-clean 证据，表明 SSD 中间层确实提供了
超越 VAE 重建细节的 held-out 几何信息。因此应继续扩展冻结判别式 probe，但不能
把结果外推为 diffusion RGB 路径恢复成功：后者仍应止损。下一阶段应做多 seed、
更大 train subset，并从 train 中划 dev 做 checkpoint selection，official val 只在
最终 endpoint 使用；VAE-only 控制组继续保留。

### 3.10 Stage 8 多 seed / dev-selected 终点评估（07-27）

Stage 8 将声明的 train manifest 固定拆为 128 张 `train-fit`、32 张内部
`dev`，并保留 32 张完全不参与选点的 `official val`；分割 checksum 为
`b674806977e5e44c37bcee81c2c23e296d7ffb2ee16d4846250a45240390a013`。seed
42/43/44 使用相同六输入 head、每 seed 相同的 VAE-only/VAE+SSD 初始化与
batch schedule、128px 和 300 steps；每个 arm 只按 dev joint score 选 checkpoint，
随后才做一次 official-val 终评。

| Seed | 选定 step（VAE-only / VAE+SSD） | official Pearson Δ | official Spearman Δ | official MAE Δ |
|---:|---:|---:|---:|---:|
| 42 | 275 / 175 | +0.0744 | +0.0797 | -0.0062 |
| 43 | 300 / 125 | +0.0595 | +0.0633 | -0.0060 |
| 44 | 300 / 150 | +0.1194 | +0.1073 | -0.0060 |
| **均值** | -- | **+0.0844** | **+0.0834** | **-0.0060** |

三 seed 的 dev Pearson/Spearman 增量均为正，且与 official-val 方向一致；三
seed 均同时改善两项相关性，超过“至少 2/3 seed”要求，聚合 MAE 也未恶化。
因此 **Stage 8 多 seed gate 通过**，冻结 SSD 判别式路径可进入更大规模验证。
这仍只证明 relative-depth 标量读出，不能恢复此前失败的 RGB diffusion、LoRA、
Adapter/ControlNet 路径，也不能宣称 metric depth。

### 3.12 全量数据 Stage 9（07-27）

随后在 `vb_snvc_train_v3` 的完整声明数据上运行 Stage 9：2,182 张 train
全部纳入协议，其中 1,926 张用于 fit、256 张用于 dev；115 张 official val
保持完全隔离，只在 dev 选点后评估。配置为 128px、4,000 steps、seed
42/43/44，三 seed 均完成 VAE-only/VAE+SSD matched 对照；全流程 MPS 用时
约 1,822.7 秒。

| Seed | 选定 step（VAE-only / VAE+SSD） | official Pearson Δ | official Spearman Δ | official MAE Δ |
|---:|---:|---:|---:|---:|
| 42 | 4000 / 3000 | +0.0762 | +0.0794 | -0.0154 |
| 43 | 4000 / 3200 | +0.0887 | +0.0931 | -0.0195 |
| 44 | 4000 / 2600 | +0.0787 | +0.0780 | -0.0159 |
| **均值** | -- | **+0.0812** | **+0.0835** | **-0.0169** |

VAE+SSD 的 official-val 聚合绝对指标为 MAE `0.1733`、Pearson `0.6527`、
Spearman `0.6345`；三 seed 均通过增量 gate。相较 Stage 8 的 128/32/32
小规模实验，扩大数据后绝对相关性明显提高，说明全量数据确实带来更好的
泛化效果。当前 checkpoint 是完整数据集实验 endpoint，但尚未做“冻结 dev
选点后、用全部 2,182 张 train 重新拟合”的最终部署 refit；因此仍应称为
full-dataset validation，而不是最终 production 权重。

### 3.11 复现完成度重新评级

| 时间 | 评级 | 依据 |
|------|------|------|
| 06-17 | D+ | 无权重、无训练实证 |
| 07-26 晨 | D+ | 独立核验：无已发布权重 |
| **07-26 晚（本报告）** | **C-** | 训练/评估闭环与 checkpoint 往返成立；单样本信号存在，但新增多样本实验表明颜色误差下降并不等于几何学习，**泛化能力与论文数字对标均未证明** |
| **07-27 凌晨** | **C-** | 三类 geometry gate 均已完成且止损；工程验证更充分，但四样本训练排序仍未被稳定记忆，尚无理由上调实证评级 |
| **07-27 Stage 6** | **C-** | 冻结特征轻量头完成四图几何记忆，证明 SSD/VAE 表征可读；仍无 held-out 证据，因此维持评级 |
| **07-27 Stage 7** | **C-** | SSD 中间特征在 16 图 held-out 上显著优于 matched VAE-only；但路径已转为判别头、绝对相关性仍中等，尚不足以上调原生成范式复现评级 |
| **07-27 Stage 8** | **C-** | 三 seed、dev-selected、official-val-only 终点评估通过 SSD 增量 gate；证据支持冻结判别式 relative-depth 路径，但不改变原生成范式复现评级 |

---

## 四、代码质量评估

### 4.1 优点

- **工程规范一流**：SemVer + Keep-a-Changelog（每条变更注明文件与动机）、ruff（E/F/W/I/N/UP/B/SIM）+ mypy + pre-commit + 3 个 CI workflow（含性能回归门禁）。
- **性能优化有纪律**：所有加速（Union-Find、scatter_add、S4 预过滤）均标注「exact, not approximate」并配回归测试；性能测试用 warmup + median 抗 CI 负载抖动。
- **防御性设计到位**：PEFT wrapper 误路由检测与绕过、torchao 量化迁移带 legacy fallback、`weights_only` 处理、checkpoint 元数据完整（rng/scheduler/optimizer/ema 全保存，fresh-process reload 像素级一致已验证）。
- **审计文化**：多份独立报告交叉核验，旧风险项逐条复查标注「已修复/已过时/仍存」。

### 4.2 问题分类

#### 🔴 P0 — 阻塞性（决定「复现」是否成立）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 1 | **基座替换未完成泛化实证**：论文数字建立在闭源 NBP 上；SSD/VAE 冻结特征已通过四图读出 gate，但 held-out 质量未知 | 全局 | 已证明局部表征可读，仍不能推导与 NBP 同等的泛化或生成能力 |
| 2 | **训练标签与论文编码不一致**：v3 数据是 viridis 而非 cube-edge 幂变换 | `data/vb_snvc_train_v3` | 即使训练成功，也无法直接与论文 cube-edge 数字对比；且 metric depth 不可恢复 |
| 3 | **作者/引用矛盾**：README 署名 "Shengbang Bang…Yann LeCun" vs 对齐报告 "Valentin Gabeur★, Shangbang Long★, Songyou Peng★"，arXiv:2604.20329v3 真伪未核验 | `README.md` vs `docs/paper_alignment_2604.20329v3.md` | 学术诚信层面的硬伤，对外发布前必须统一 |
| 4 | **无已发布权重**：checkpoint 注册表全为 placeholder | `configs/checkpoint_registry.yaml` | 社区无法复现任何推理结果 |

#### 🟡 P1 — 显著（本周内修复）

| # | 问题 | 位置 | 影响 |
|---|------|------|------|
| 5 | MPS 训练吞吐 74 s/step（512px），单轮实验 42 小时 | 训练日志实测 | 迭代节奏被硬件锁死；建议 H20/A100 配置已有但未作为主路径 |
| 6 | `model_registry.py` 单文件 2221 行，覆盖 23 家族 | `model_registry.py` | 超出健康单文件规模 ~4×，应按 family 拆分（sd/flux/dit 子模块） |
| 7 | Gemini filtering 未实现，instance/referring 评估闭环不完整 | `evaluation/instance.py` | cgF1/IL_MCC 不可信 |
| 8 | full-pipeline ONNX 仍依赖 `getattr(gen,"_t2i_pipe",None)` 兜底，报错文案与实际检查不符 | `export/onnx_export.py` | 导出路径脆弱 |
| 9 | specular_removal 训练无 val_dataset，per-step 评估被禁用（日志确认） | `ssd_specular_focused_train.log:24` | 2044 步训练盲跑，无过拟合预警 |

#### 🟢 P2 — 优化项（本月规划）

| # | 问题 | 位置 |
|---|------|------|
| 10 | 5 个 speculative 任务（edge/flow/keypoint/detection/depth-edge）无验证基准 | README 任务表 |
| 11 | README benchmark 表虽标注 "from paper" 但位置易误读，需顶部醒目声明「本仓库未复现」 | `README.md` |
| 12 | INT8 量化对 RGB 映射精度的系统评估缺失（roundtrip 误差对 8-bit 敏感度） | `inference/optimization.py` |
| 13 | `weights_only` 不一致（建议统一 True 或迁移 safetensors） | `training/checkpoint.py` / `inference/checkpoint.py` |

---

## 五、差距分析与关键瓶颈

| 维度 | 评分 | 状态 |
|------|------|------|
| 学术对齐度 | A- | 核心算法与论文高度一致；基座与数据编码两处实质偏离 |
| 工程完成度 | B+ | 分层/测试/CI/文档齐全；registry 偏大、ONNX 兜底脆弱 |
| 实证进展 | C- ↑ | 闭环跑通、单样本信号确认；泛化未证明（本次上调） |
| 文档可信度 | B | 详尽且诚实（负结果完整记录）；作者矛盾 + 数字易误读待修 |
| 生产就绪度 | C+ | serving/Docker/preset 齐备；量化与吞吐未验证 |

**关键瓶颈排序**：① FLUX/SSD 基座能否接近 NBP 表征质量（科学问题）→ ② 训练数据编码与论文的一致性（数据问题）→ ③ MPS 吞吐导致的迭代速度（工程问题）→ ④ instance/referring 的外部模型依赖（闭环问题）。

---

## 六、路线图建议

### Phase 1（本周）— 让「复现」二字站得住
1. **完成 specular 训练后，立即在 135 样本官方 val split 上评估**（工具已就绪：`val_relative_depth_*.json` 流程），给出 split-clean 的第一个真实泛化数字。
2. **统一论文引用与作者署名**；README 顶部加醒目声明「benchmark 为原论文 NBP 数字，本仓库开放基座版复现进行中」。
3. **决定 depth 数据路线**：要么按论文 cube-edge 幂变换重新生成 v3 标签，要么明确将目标改为「viridis 相对深度」并更新所有 prompt/manifest/评估口径——不能两者混用。
4. 发布**第一个权重**（哪怕只 depth 单任务 LoRA）到 HF Hub（`scripts/publish_to_hub.py` 已就绪）。

### Phase 2（两周）— 规模化验证
5. 迁移训练主路径到 H20/A100（`configs/training_h20_*.yaml` 已备），MPS 仅作 smoke；目标吞吐 ≥10×。
6. 按论文 9:1 数据混合比例跑 multi-task LoRA（depth + normal + semantic），在 Hypersim/ScanNet/Cityscapes 真值数据上对标论文数字——这是唯一能回答 P0-1 的实验。
7. 拆分 `model_registry.py`（按 family 子模块化），补齐其真实覆盖率。
8. 实现或替代 Gemini filtering（可用开放 VLM 如 Qwen2-VL 替代做负样本过滤）。

### Phase 3（月度）— 走向生产与社区
9. speculative 任务要么给验证基准，要么降级标注「实验性」。
10. INT8/INT4 量化对 roundtrip 精度的系统评估；视频时序一致性扩展。
11. 修复 ONNX full-pipeline 兜底逻辑与报错文案。

---

## 七、结论

这是一个**工程质量明显高于同类「论文复现」仓库的范式验证框架**：算法层与论文逐式对齐且实测可逆，训练/评估/导出/服务全链路齐备，审计文化诚实（负结果完整记录、旧报告逐条复查）。

它的当前真实状态是：**「脚手架已验收，实证攻坚刚打出第一缕正向信号」**——单样本几何学习确认（标定后 r=0.935），但 split-clean 泛化、metric depth、论文数字对标三件事都还没有发生。对外宣称「复现 Vision Banana SOTA」目前不成立；宣称「正在以工程级严谨度系统性验证该范式」则完全成立。

**总评级：B（脚手架 A- / 实证 C-，处于明确上升通道）。** 下一步的优先级无可争议：Phase 1 的四件事做完之前，不要碰任何新功能。

---

*本报告基于 2026-07-26 17:25 前的源码、文档与 `output/` 实测数据撰写；训练仍在进行，结论可能随后续 checkpoint 评估更新。*
