# Vision Banana 复现对齐审计

日期：2026-10-09。范围：当前工作区源码（含未提交修改）、配置、历史报告，以及 arXiv:2604.20329v3 原文。

## 结论

项目已经实现论文主要 RGB 编码/解码、训练和评测框架，但尚未证明“同一图像生成模型经少量视觉任务对齐后，获得跨任务零样本能力且保留生成能力”。

准确定位是：**开源替代基座的方法复现与范式验证项目，而非论文数值复现完成品**。不建议继续使用“代码 100% 对齐”“训练验证 40%”等没有可检验分母的完成度。

最重要的下一步不是扩展任务或直接增加训练步数，而是先保证配置实际生效、诊断误差来源，再通过固定协议的单任务结构闸门，最后推进共享权重混训。

## 证据边界

- 当前工作区有大量既有未提交修改，本次没有修改这些文件。
- 仓库内未发现可供复算的 `data/`、`output/`、模型权重或原始指标产物。历史报告中引用的路径不代表当前机器拥有这些资产。
- 本机 `python3` 缺少 torch 和 pytest；本次没有执行训练、真实模型推理或测试套件。
- 2026-08-31 的 `1031 passed, 15 skipped` 是历史记录，不能作为当前源码的测试结果。
- 原文通过 arXiv v3 HTML 核对；本地 PDF 未成功提取文本。本文不判断是否存在更新版本。
- 对配置解析函数进行独立抽取执行，确认论文基线实际得到 `pair_manifest=None`、`data_dir=None`、`lr=3e-5`、`steps=5000`、`grad_accum=4`、`output_dir=output/training`；字符串权重同样实测回退为默认字典。这是局部函数检查，不是完整 Trainer 测试。
- `git diff --check` 通过；没有训练代码改动。

## 原论文究竟要求什么

原文：[Image Generators are Generalist Vision Learners](https://arxiv.org/html/2604.20329v3)。

1. 使用 Nano Banana Pro，以原生成数据和低比例视觉任务数据进行 instruction-tuning。
2. 视觉输出均为可解码 RGB；同一模型、同一套权重，通过指令切换任务。
3. 不添加任务专用架构，不依赖定制任务损失来支撑核心结果。
4. 3D 对齐数据来自渲染；深度明确完全合成，不使用真实世界深度训练数据。
5. 对齐训练不包含评估 benchmark 的训练集；这里的 zero-shot transfer 指没有使用这些 benchmark 的训练集，不等于未经任何视觉任务微调。
6. 验证 2D/3D 理解，同时用人类配对评估验证文生图和编辑能力保留。

论文未披露具体混合比例、更新参数范围、LoRA rank、学习率、训练步数、训练数据量及完整采样/评测协议。项目中的 LoRA、Min-SNR、flow bridge、10% 视觉比例均应标为实现选择，不应写成论文公开参数。

“lightweight”不能推出“论文使用 LoRA”，也不能推出“没有更新全部参数”。原文批评的是不保留生成混合数据、牺牲通用性的专门微调路线。

## 对齐现状

### 已有方法实现

- 深度：Eq.1，lambda=-3、c=10/3，八顶点七段 cube-edge 插值和最近线段逆映射。见 `src/open_vision_banana/core/depth_mapper.py`。
- 法线：相机坐标系及 R=(1-x)/2、G=(1+y)/2、B=(1+z)/2，解码归一化。见 `src/open_vision_banana/core/normal_mapper.py`。
- 分割：语义最近色；实例多阶段聚类与论文参数。见 `src/open_vision_banana/core/segmentation/decoder.py`。
- 基准：度量/尺度对齐诊断、法线角误差、语义混淆矩阵、RefCOCOg，非有限值严格失败及 native-GT 计分。
- 新增开放词表语义 manifest：逐样本类别/颜色/背景/表达式变体。是数据和指令能力，尚非模型开放词表能力的证据。
- 新增 Kontext 独立图像上下文路径，已有随机小模型 token/时间/速度符号测试，尚无真实 12B 权重收敛结果。

### 未闭环的核心能力

- 主统一训练入口未接通低比例生成混训。
- 没有可核验的同一共享 checkpoint 的五任务 benchmark 结果。
- 没有当前共享 checkpoint 的文生图/编辑保留结果。
- 缺少完整 SA-Co/Gold 的 NP 正负查询与存在性过滤流程、ReasonSeg 推理转写流程，以及 nuScenes 深度入口。
- 相对深度 specialist、冻结特征 probe、Tiny CNN 是独立对照，不能替代生成式共享模型的复现证据。

## 当前源码中的阻断项

### P0：配置声明与实际执行不一致

`configs/paper_reproduction_baseline.yaml` 使用 `data.root`、`optimizer.lr`、`training.epochs`、`gradient_accumulation_steps`、顶层 `output_dir`。统一解析器读取 `data.dir/pair_manifest`、`training.lr/steps/gradient_accumulation`、`checkpoint.output_dir`。

直接使用此配置会缺少 manifest；学习率、累计步数等静默退回默认值。配置里的 generation_path、task_sampling 等也未被统一解析器消费。证据：`training/config.py` 的 `_load_yaml_config()`，`training/trainer.py` 的 `_build_datasets()`。

验收：所有生产配置走同一 schema；未知字段硬失败；在加载模型前输出并检查 resolved config、数据计数、实际 trainable module 列表。

### P0：vision_ratio 不等于已执行混训

统一 Trainer 只拼接视觉任务 manifest，未加载独立生成数据集或调用 MixedVisionDataset。配置 `vision_ratio=0.1` 不能证明采样中实际只有 10% 视觉数据。

验收：按优化步记录视觉/生成实际比例、各任务采样比例，以及生成样本的正确条件路径。纯文生图样本不能未经区分地沿用“源图到目标图桥接”目标，否则可能成为重建任务而非生成 replay。

### P1：任务权重与 LoRA targets 可能失效

- `TrainingConfig.parse_task_weights()` 优先返回默认非空字典，字符串配置不会覆盖默认 depth=2.5、normal=2.0、semantic=0.5。
- `ModelBackend.__init__()` 接收 `lora_targets`，但 `apply_lora()` 未使用这个 override。

验收：从 YAML 到实际损失权重、可训练参数名的端到端检查，而不仅检查字段存在。

### P1：模型选择口径不稳

验证只看拼接数据前 20 个 batch 的 flow loss；数据按任务排序，可能只覆盖 depth。验证使用 EMA，恢复原权重后才触发 best 保存，best adapter 不必对应被验证的 EMA 权重。

验收：按任务分层、固定样本/噪声验证；明示评测原权重还是 EMA；保存与评测完全相同的权重；以开发集解码指标辅助选优，而非只看 flow loss。

### P1：失败样本与溯源不够严格

manifest 缺文件会被过滤，图像读取失败会替换随机候选；空数据或不足一个 batch 还可能产生空 loader。checkpoint 没有形成完整配置、Git 状态、数据/编码指纹和 RNG 的统一留档。

验收：正式实验不静默丢样本或替换；训练启动前验证非空 loader；保存固定 manifest 内容指纹、权重 hash、有效配置、协议、随机状态和所有失败记录。

## 历史结果该如何解释

来源：`docs/plans/2026-08-31-acceptance-stage-progress.md`。

- Hypersim metric-depth specialist：NYU n=32，delta1=0.3578，AbsRel=0.4367；旧 n=654 delta1=0.4030。
- Hypersim+KITTI replay：NYU n=32 delta1=0.2859，触发停止。
- Cityscapes specialist：训练用 Cityscapes，50 张验证图 mIoU=0.0300。只能作为域内学习诊断，不是 zero-shot transfer。
- normal：缺官方 target，未晋级。
- 当时共享 LoRA 和新共享模型的生成 replay 尚未启动。

**这些深度绝对数不能拿来计算今天的论文差距。** `docs/paper_alignment_2604.20329v3.md` 附录 B 已说明旧 NYU crop 错误、KITTI crop 未生效。必须对同一旧 checkpoint 按修复后协议重新跑模型和基线。

原论文目标：NYU delta1=0.948、AbsRel=0.081；Cityscapes mIoU=69.9%；四法线集 mean=18.928 度。它们是远期参照，不是当前结构闸门。

## 进一步对齐中最关键的纠偏

### 1. Kontext 是路线可行性实验，不是条件机制的因果消融

`configs/experiments/flux_kontext_depth_gate.yaml` 同时改变基座、规模、预训练、条件方式，并声明更改学习率/rank，且有量化配置。相对旧 SANA 的好坏无法归因于单一条件机制。

- 成功：证明该替代基座在给定预算下可学会该任务。
- 失败：只证明该配置/数据/预算组合未通过，不能推出“主因必然是数据规模”。
- 因果判断：需要尽可能同一基座、同一数据和训练预算的合法条件路径对照；无法控制时明确写成跨基座路线比较。

### 2. 度量深度首先要排查表示链条

Eq.1 在该参数下为 u=1-(1+d/10)^(-2)，逆变换为 d=10*((1-u)^(-1/2)-1)。远处 u 接近 1，微小色彩误差会被放大；终点白色对应无穷远，必须报告饱和/截断处理。

训练前依次测：真实米制 GT -> 浮点 RGB -> 8-bit RGB -> VAE encode/decode -> RGB 解码。各阶段按深度距离分桶报告 AbsRel、delta1、无效像素/饱和比例。模型未训练就丢失的信息不能靠加步数解决。

同时审计 Hypersim 提供的是沿光线距离还是相机平面 z-depth。两者若需要转换，允许预处理使用渲染相机信息，但不得将相机参数输入预测模型；需明确记录此区别。

### 3. 开放词表必须用反事实指令检验

同一图像交换类别颜色、增加不存在类别、改变背景色、改写 prompt；语义 mask 应基本不变而 RGB 跟随映射变化。固定色表准确率不能证明模型理解指令。

新 manifest 支持的只是给定类别表内的变化；真正未见词语/短语泛化还需独立查询集合。先在可控开发数据过“读图例”闸门，再报告 Cityscapes 零样本 mIoU。

### 4. 训练源和评估源不能相互污染

旧 roadmap 推荐 ScanNet normal 训练，同时把 ScanNet 用作 benchmark；Virtual KITTI 训练也会影响同数据源的 normal 零样本声明。严格轨道应排除评估源的训练集/同场景资产，跨任务输入复用也要审计。

可以保留 KITTI/Cityscapes specialist 做工程上限，但单独归档为域内诊断，不能混入论文零样本主结果。测试集不得用于超参数或 checkpoint 选择。

### 5. 外部流程成绩要拆开报

SA-Co/Gold 47.5 是 Vision Banana + Gemini 3.1 Flash-Lite，后者做 NP 存在性过滤。ReasonSeg 79.3 是 Vision Banana + Gemini 2.5 Pro，后者先转写推理查询，单轮调用。

换成 Qwen 等开源模型是合理替代实验，但不是相同系统。分别报告生成器本体、外部模型、组合系统、调用预算和时延。FID/LPIPS 可补充，但不能替代论文生成/编辑人类配对胜率。

## 下一阶段建议与验收顺序

1. **固定有效实验契约**：修复配置失效、数据静默替换、权重选择；生成 resolved config 和审计 manifest。没有通过，不启动昂贵训练。
2. **恢复证据资产**：定位旧 checkpoint/原始指标/数据；按修复后同协议重新评测，并报告像素加权与逐图平均、native GT 和 crop 假设。
3. **最小深度定位实验**：先做表示链条 oracle，再做小样本过拟合；开发集加入打乱输入图、打乱/屏蔽指令、常数/训练均值图对照。要证明模型利用空间内容，而不只是估计平均尺度。
4. **编辑基座单任务闸门**：保留 Kontext 候选；按 `evaluation/gates.py` 比最佳非 oracle delta1 至少高 0.10，AbsRel 不劣于最佳非 oracle，尺度对齐后超过逐图中位数 oracle。阈值是项目自定，不是论文指标；晋级需独立测试确认。
5. **normal 与动态语义闸门**：官方 normal GT、坐标约定核验；semantic 颜色交换一致性和完整图例。共享训练前先通过单任务闸门，是项目的风险控制策略，不是论文规定。
6. **共享权重+生成混训**：禁止拼接各任务不同最佳 adapter 充当统一模型。基于固定共享 checkpoint，扫描视觉采样比例；可先探索 1%、5%、10%、纯视觉对照，固定视觉曝光量并同时报告总计算量，不能称作论文比例。
7. **最终统一验收**：完整五任务、外部组合流程和配对生成/编辑评估；至少多随机种子或相应不确定性估计。提前定义可接受的生成能力退化界限，不能以“与 50% 不显著不同”直接宣称能力相等。

## 对外可用表述

当前：“已实现 Vision Banana 的主要 RGB 任务表示、开源生成式训练框架与部分评测协议；真实共享模型的跨任务零样本能力及生成能力保留尚待验证。”

达到下一阶段后：“在开源编辑基座上通过了固定协议的单任务空间结构与尺度闸门。”

只有同一模型完成多任务、无 benchmark 对齐训练污染并通过生成能力保留评测后，才适合声称验证了论文核心范式。即便如此，仍应与闭源 NBP 的数值复现明确区分。
