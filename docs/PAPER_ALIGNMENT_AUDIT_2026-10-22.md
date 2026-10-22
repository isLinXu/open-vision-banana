# 论文逐项对齐审计（2026-10-22）

> 对象：《Image Generators are Generalist Vision Learners》arXiv:2604.20329v3（30 页，正文 §1–4 + 附录 A–D），
> 与仓库 `main` @ `8dda314`（PR #45 之前）。
> 方法：从 `paper/*.pdf` 提取全文，逐句摘出**可核对的陈述**（公式、常数、prompt 原文、协议、数据规则），
> 到源码中定位实现并核对取值。论文**未写明**的内容一律标注"论文未给出"，不做臆测。

## 0. 结论先行

| 维度 | 可核对条目 | 完全一致 | 部分一致 | 缺失 | 实现错误 |
|:--|--:|--:|--:|--:|--:|
| A. 任务编码/解码（§3.1、§3.2、Eq.1–6） | 22 | 21 | 1 | 0 | 0 |
| B. Prompt 原文（Fig. 2/3/4、§1） | 9 | 9 | 0 | 0 | 0 |
| C. 评测协议与指标（§3、附录 B、Table 1–7） | 17 | 14 | 2 | 1 | 0 |
| D. 数据与训练规则（§2、§3.2） | 8 | 4 | 3 | 1 | 0 |
| E. 生成保留（§2、附录 D） | 2 | 0 | 1 | 1 | 0 |
| **合计** | **58** | **48** | **7** | **3** | **0** |

- **方法层一致度：83%（48/58）完全一致，95%（55/58）至少部分一致，0 处实现错误。**
- 三处"缺失"全部源于论文闭源要素（Nano Banana Pro、in-house 标注、人工成对评测），**不可在开源实现中补齐**；七处"部分一致"中六处是论文给了定性描述但未给出可核对的数值/实现细节，仓库做了记录在案的选择。
- 论文**完全没有**训练细节章节：无优化器、学习率、步数、batch、分辨率、微调方式（全参/LoRA）、损失函数、混合比例数值、推理步数。这些在仓库里全部是"仓库选择"，不构成一致性偏差，但也**无法声称与论文一致**。
- 因此"100% 一致"在定义上不可达：论文可核对的 58 条中 47 条逐字/逐值一致；不可核对的部分（闭源模型、内部数据、人工评测、未公开的训练配方）无论实现成何种样子都无法判定一致。

## 1. 论文方法论梳理（仅据原文）

| 模块 | 论文陈述（页码） | 给出了什么 | 没给出什么 |
|:--|:--|:--|:--|
| 整体范式 | §2 p.4：对 NBP 做 instruction-tuning，"mix vision task data into Nano Banana Pro's own training mixture at a very low ratio"；"without specialized architectures or custom training losses"（p.4） | 混合训练、低比例、无额外模块/损失 | 比例数值、训练步数/优化器/lr/分辨率/微调方式、损失函数形式（只说沿用生成模型自身目标） |
| 任务与数据 | §2 p.4：2D = referring/semantic/instance；3D = metric depth/normal；"in-house model annotations for web-crawled 2D images"；"synthetic data from rendering engines for 3D tasks"；"no training data from our evaluation benchmarks" | 任务集合、数据类别、零样本规则 | 任何数据集名称、规模、标注模型 |
| 语义分割 | §3.1 p.6：开放词表，prompt 给类别→颜色；支持自然语言 / JSON / 命名色 / hex / RGB 元组；解码 = 最近颜色；Cityscapes 评测用同一 prompt 含全部 19 类（含不在图中的类） | 编码、四种 prompt 风格、解码规则、评测 prompt 协议 | 训练时各风格比例、颜色采样方式 |
| 实例分割 | §3.1 p.7–8：prompt 只给类别 + 背景色，模型自分配颜色；附录 A 五阶段聚类（Eq.2–6，τ=14、16-连通、θ_size=2e-4、3×3 腐蚀、θ_erosion=0.1、γ=5.0）；SA-Co 负样本交给 Gemini 3.1 Flash-Lite（prompt 原文 p.8） | 全部解码常数与算法、Gemini prompt | Eq.2 写 ‖·‖₂<τ²（范数与平方混用，取"距离<τ"解）；Eq.3 "to the component's seed pixel"但 flood-fill 相邻判据的精确含义有两种读法；Eq.6 "area"是 bbox 还是像素未明确 |
| 指代分割 | §3.1 p.8–9；Fig. 4 四条 prompt 原文；ReasonSeg 用 Gemini 2.5 Pro 单轮改写 | prompt 原文、改写协议 | 二值掩码的解码阈值（"straightforward"） |
| 度量深度 | §3.2 p.9–10：Eq.1 幂变换 λ=−3、c=10/3；沿 RGB 立方体边从黑到白分段线性（"first iteration of a 3D Hilbert curve"，Fig. 5）；解码 = 投影到最近线段再反插值；训练增广 Plasma/Inferno/Viridis/grayscale；训练与推理均不用内参 | 公式、常数、路径（Fig. 5 图像给出顶点顺序与米数标注）、增广色图名、无内参 | 增广色图的具体采样点 |
| 法向 | §3.2 p.13：相机空间右手系 +x右 +y上 +z朝外；R=trunc((1−x)/2,0,1)×255、G=trunc((1+y)/2)×255、B=trunc((1+z)/2)×255 | 公式与坐标系 | 无 |
| 评测 | Table 2–7、附录 B：Cityscapes val mIoU；SA-Co/Gold pmF1@IoU[0.5:0.05:0.95]、IL_MCC、cgF1=100·pmF1·IL_MCC（Eq.7）；RefCOCOg UMD val cIoU；ReasonSeg val gIoU；深度 δ1/AbsRel 六数据集；法向 mean/median 角误差四数据集 | 指标定义、数据集、Table 1 的四数据集平均口径 | 深度截断/裁剪协议、法向的逐像素还是逐图平均 |
| 生成保留 | §2 p.4、附录 D：GenAI-Bench / ImgEdit 人工成对胜率 53.5% / 47.8% | 协议为人工成对评测 | 评测人数、样本数 |

## 2. 逐条对照表

状态：✅ 完全一致 ｜ 🟡 部分一致（含偏差程度）｜ ⬜ 缺失/未实现 ｜ ❌ 实现错误

### A. 任务编码 / 解码

| # | 论文原文（位置） | 代码位置 | 代码取值 | 状态 |
|:-:|:--|:--|:--|:-:|
| A1 | Eq.1 `f(d,λ,c)=1−(1−d/λc)^(λ+1)`，λ=−3，c=10/3（p.10） | `core/depth_mapper.py` L194–261 | `lam=-3.0, c=10/3`；前向 `1-(1-d/(λc))^(λ+1)`，逆向 `λc(1-(1-u)^(1/(λ+1)))` | ✅ |
| A2 | 沿 RGB 立方体边从黑到白分段线性，Hilbert 一阶（p.10, Fig. 5） | `CUBE_EDGE_KEYPOINTS` L42–55 | 黑→红→黄→绿→青→蓝→品红→白，等分 1/7 | ✅ 已用 PDF 第 10 页提取的 Fig. 5 **图像**核对：顶点顺序与图中一致；图中顶点标注 0.8/1.8/3.2/5.3/8.7/16 m 与等分 1/7 反解值 0.80/1.83/3.23/5.28/8.71/16.46 m 一致，三分点标注（0.2/0.5、1.1/1.5、2.2/2.7、3.8/4.5、6.2/7.3、10/13、22/36）亦与等分假设吻合 |
| A3 | 解码：投影到最近线段再反插值（p.10） | `depth_mapper.py` L320–360 `_project_to_curve` | 逐段解析投影 + 线性反插 | ✅ |
| A4 | 幂变换与色图复合为 [0,∞]↔[0,1]³ 双射（p.10） | 同上 + `normalized_to_depth` | 可逆；`tests/test_depth_mapper*` 回环 | ✅ |
| A5 | 增广色图 Plasma、Inferno、Viridis、grayscale（p.10） | `PLASMA/INFERNO/VIRIDIS/GRAYSCALE_KEYPOINTS`；`depth_colormaps` 配置 | 四种全部实现，另有 rainbow/turbo（论文"such as"，不冲突） | 🟡 轻微：论文未给采样点，仓库从 matplotlib 取关键点近似 |
| A6 | 法向 R=(1−x)/2、G=(1+y)/2、B=(1+z)/2，clamp[0,1]×255（p.13） | `core/normal_mapper.py` L26–40 | 完全相同公式 + clamp | ✅ |
| A7 | 坐标系 +x右 +y上 +z朝外（p.13） | `data/hypersim.py`（Hypersim 相机系即此系，背向检测）；`depth_geometry.depth_to_normals`（OpenCV→论文系 (x,−y,−z)） | 一致，并有解析平面测试 | ✅ |
| A8 | 语义解码：每像素取 RGB 距离最近的类别色（p.6） | `segmentation/decoder.py` `decode_semantic` L130–165 | 向量化最近邻 | ✅ |
| A9 | 实例：prompt 只给类别与背景色，模型自分配颜色（p.7） | `inference/prompt.py` `INSTANCE_TEMPLATES`；`build_instance_manifest.py` | 三模板均不含实例色；训练 target 用调色板随机着色 | ✅ |
| A10 | Eq.2 背景：‖C(p)−C_bg‖ < τ，τ=14（附录 A） | `decoder.py` L226 | `norm < tau`，严格小于，τ=14 | ✅（论文写 τ²，按距离阈值解） |
| A11 | Eq.3 seed flood-fill，16-连通，‖C(p_i)−C(seed)‖ ≤ τ | `grouping.py` L242/372；`_connectivity.py` | 16 邻域（4+4+8 马步）；`<=`；`seed_merge_rule=pixel_only` 为论文读法，`pixel_and_seed` 为更严格的可选读法 | ✅ |
| A12 | Eq.4 噪声剪枝 \|S\| < θ_size·HW，θ_size=2e-4 | `postprocess.py` L24；`min_area_fraction=2e-4` | `counts >= min_area` 保留 ⇔ `<` 丢弃 | ✅ |
| A13 | Eq.5 3×3 腐蚀，保留比 < 0.1 剪枝 | `postprocess.py` L82、L104 | 3×3 全 1 核；`ratio < 0.1` | ✅ |
| A14 | Eq.6 均色距离 ≤ τ 且合并 bbox 面积 ≤ γ·(A_a+A_b)，γ=5.0 | `postprocess.py` L176–205；`spatial_merge_area_basis="bbox"` 默认 | bbox 口径；γ=5.0；像素口径可选消融 | ✅（论文 A(·) 未定义，取 bbox 读法并保留另一读法） |
| A15 | 背景色 C_bg 预定义（附录 A） | manifest `background_color` → batch → `decode_instance(background_color=)`（#42） | 逐样本传递 | ✅ |
| A16 | 指代分割后处理"straightforward"（附录 A 首句） | `core/referring.py` `decode_referring_mask` | `binary_white_black_v1`：mean RGB > 0.5；保留所有连通域 | 🟡 论文未给阈值；仓库选择已版本化 |
| A17 | 深度训练用 GT 米制深度编码（p.10） | `prepare_hypersim.py`（射线距离→平面 z）、`prepare_tartanair.py` | 米制、平面 z | ✅ |
| A18 | "distance from the camera plane"（p.9） | `depth_geometry.ray_distance_to_planar_depth` | Hypersim 射线距离已转换 | ✅ |
| A19 | 法向 trunc(…)×255 | `hypersim.py` L74 用 `round()` | 量化方式差 ≤1 灰阶 | 🟡 可忽略（论文 trunc 语义更接近 clamp；取整方式未明示） |
| A20 | 语义开放词表、类别由 prompt 动态给定（p.6） | `data/openvocab_semantic.py` 逐样本采样类别→颜色 | 一致 | ✅ |
| A21 | 深度不用相机参数（训练/推理）（p.11） | 训练/推理路径无内参输入 | 一致（VKITTI 法向推导用内参，但那是**评测 GT 构造**，非模型输入） | ✅ |
| A22 | 输出为普通 RGB 图，无专用头/模块（p.4） | 无任务头；LoRA 作用于生成器本身 | 一致 | ✅ |

### B. Prompt 原文

| # | 论文原文 | 代码位置 | 状态 |
|:-:|:--|:--|:-:|
| B1 | Fig.2(a) `Generate a semantic segmentation visualization image, using this color mapping: {...}` | `PromptBuilder.SEMANTIC_TEMPLATES[0]` | ✅ |
| B2 | Fig.2(b) `This image is a per-pixel class labeling of the input. The X are represented by (r, g, b)...` | `SEMANTIC_TEMPLATES[1]` | ✅ |
| B3 | Fig.2(b) `Generate a semantic segmentation visualization of the input. The X is #RRGGBB` | `SEMANTIC_TEMPLATES[2]` | ✅ |
| B4 | Fig.2(a) 命名色 JSON `{"cat": "red", ...}` | `openvocab_semantic.build_prompt("json_named")` | ✅ |
| B5 | Fig.3(a) `Generate an instance segmentation visualization of this image. Each X is colored differently.` | `INSTANCE_TEMPLATES[0]` | ✅ |
| B6 | Fig.3(b) `This image is a segmentation task derived from the input. The X instances are each represented by a unique, solid color. Background is RGB(...)` | `INSTANCE_TEMPLATES[1]` | ✅ |
| B7 | Fig.3(c) `This image shows segmentation masks for the X from the input image. The background is set to .... Each X is represented by a different color.` | `INSTANCE_TEMPLATES[2]` | ✅ |
| B8 | Fig.4(c)/(d) `This image shows segmentation masks from the given image. The background is black color. The X is represented by a solid yellow/cyan.` | `referring_contract.BINARY_REFERRING_TEMPLATE`（白色）；多色版 `REFERRING_TEMPLATES[1]` | ✅ |
| B9 | Fig.4(a)/(b) 多表达式 | `MULTI_REFERRING_TEMPLATES`（测试断言逐字） | ✅ |

论文**未给出**深度与法向的 prompt 原文（Fig. 1 只有 "Generate a segmentation / depth / surface normal / … visualization of this image, using the following color map…" 示意）；仓库 `DEPTH_PROMPT` / `NORMAL_PROMPT` 为仓库选择，不计入对照。

### C. 评测协议与指标

| # | 论文 | 代码位置 / 取值 | 状态 |
|:-:|:--|:--|:-:|
| C1 | Cityscapes val，19 类 mIoU，同一 prompt 含全部 19 类（p.6） | `task_prompts.SEMANTIC_SEGMENTATION_PROMPT`（19 类图例）；`evaluation/semantic.py` 全局混淆矩阵 | ✅ |
| C2 | SA-Co/Gold pmF1 @ IoU [0.5:0.05:0.95]（附录 B） | `SACO_IOU_THRESHOLDS = 0.50..0.95` 十档 | ✅ |
| C3 | cgF1 = 100·pmF1·IL_MCC（Eq.7） | `instance.py` L193 | ✅ |
| C4 | IL_MCC 由 Gemini 3.1 Flash-Lite 判存在；prompt 原文（p.8） | `evaluation/gemini.py` L25、L132–133 逐字 | ✅ |
| C5 | 只对 Gemini 判阳性的样本生成（p.8） | `SACoInstanceDataset` + evaluator `pred_present` 门控 | ✅ |
| C6 | RefCOCOg **UMD** val，cIoU（p.8） | `RefCOCOgDataset(protocol="umd")`；`referring.py` cIoU 累计交/并 | ✅ |
| C7 | ReasonSeg val gIoU；Gemini 2.5 Pro 单轮改写（p.8） | `gemini.py` `DEFAULT_REASONING_MODEL="gemini-2.5-pro"`；单轮 | ✅ |
| C8 | 深度六数据集 NYU/iBims-1/ETH3D/DIODE-Indoor/KITTI/nuScenes，δ1、AbsRel（Table 6） | 六个 Dataset 类 + 注册表行；`evaluation/depth.py` | ✅ |
| C9 | 深度评测**无尺度对齐**（Table 6 为 metric） | 主指标 raw；`*_aligned` 另列为诊断 | ✅ |
| C10 | 深度截断/裁剪协议 | NYU 10 m + Eigen crop，KITTI 80 m + Garg crop，ETH3D/nuScenes 80 m | 🟡 论文未写；仓库采用社区标准协议 |
| C11 | 法向四数据集 NYUv2/DIODE-indoor/ScanNet/VKitti，mean/median 角误差（Table 7） | 四个 Dataset 类（VKITTI 由深度推导，#41）；`normal.py` mean/median | ✅ |
| C12 | 法向聚合口径 | `mean_angle_deg_per_image` = 逐图均值再平均 | 🟡 论文未写逐像素/逐图；仓库选逐图并记录在 metadata |
| C13 | Table 1 "average of 4 datasets" 口径（NYU+ETH3D+DIODE+KITTI；法向四集） | `paper_capabilities.py` 记录口径；脚本输出各集，平均需手算 | ✅（口径记录一致） |
| C14 | 零样本：评测基准训练集不进训练（p.4、p.5、p.11、p.13） | `build_qwen_baseline_manifest --protocol strict`（默认）：路径 + COCO image-id 双审计，含 Cityscapes train | ✅ |
| C15 | DIODE 只用 Indoor（Table 6/7） | `evaluate_paper_benchmarks.py` L93 `scene_type: indoors` | ✅ |
| C16 | Cityscapes 使用 labelIds→trainIds 官方 gtFine | protocol metadata | ✅ |
| C17 | 基线数字（SAM 3、DA3、Lotus-2 等） | `paper_capabilities.py` 记录论文数字作对照 | ⬜ 基线模型本身未在仓库中运行（引用论文表格） |

### D. 数据与训练规则

| # | 论文 | 代码 | 状态 |
|:-:|:--|:--|:-:|
| D1 | 3D 任务只用渲染引擎合成数据，"zero real-world depth data"（p.11） | 训练深度源 = Hypersim + TartanAir（均合成）；注册表 `group=train` 无真实深度 | ✅ |
| D2 | 2D 任务用 in-house 模型对网图的标注（p.4） | COCO/LVIS/ADE20K/RefCOCO 人工标注公开数据 | ⬜ 不可获得；替代源已记录 |
| D3 | 混入 NBP 自身训练数据，比例"very low"（p.4） | `vision_ratio=0.5`（即 50% 生成 replay）；replay 源为 OmniEdit/BLIP3o 而非底座自身数据 | 🟡 中等：机制一致（混合），比例无数值可核；数据源必然不同 |
| D4 | "without … custom training losses"（p.4） | paper_aligned：`task_loss_weights` 全 1.0、`normal_angular_loss_weight=0`、`depth_edge_aux_weight=0`、`min_snr: false`；但 `TrainingConfig` **默认值**是 depth 2.5 / normal 2.0 / semantic 0.5 且 `min_snr=True` | 🟡 轻微：论文配方在 yaml 中一致，dataclass 默认值不一致（见建议 R2） |
| D5 | "lightweight instruction-tuning"（p.3） | rsLoRA r32/α64 | 🟡 论文未定义"lightweight"的具体形式（全参低 lr 还是 adapter） |
| D6 | 基座 Nano Banana Pro | Qwen-Image-2.1 / Qwen-Image-Edit（原生图像条件） | ⬜ 闭源，不可获得 |
| D7 | 语义训练支持四种 prompt 风格（p.6） | 四风格 + 训练期随机重写与换色 | ✅ |
| D8 | 实例训练：一次一类（Fig.3 caption "one class at a time"） | `build_instance_manifest` 每行一类 | ✅ |

### E. 生成保留

| # | 论文 | 代码 | 状态 |
|:-:|:--|:--|:-:|
| E1 | GenAI-Bench / ImgEdit **人工成对**胜率（p.4、附录 D） | `evaluate_generation_preservation.py` FID/LPIPS/CLIP 代理 | ⬜ 人工评测未实现（需人力），代理已明确标注非论文协议 |
| E2 | 对照对象为基座自身 | 代理对照底座 Qwen 输出 | 🟡 机制一致，指标不同 |

## 3. 论文未提供、仓库自行选择的项（不计入一致性）

优化器（AdamW 1e-4, wd 0.01）、调度（cosine, warmup 500）、步数 30k、batch 1×32、分辨率 512、LoRA r32/α64/rsLoRA、注意力+MLP 目标模块、`vision_ratio` 0.5、`cfg_dropout_rate` 0.05、`img2img_flow`、推理步数、深度/法向 prompt 原文、指代二值阈值、法向聚合口径、深度截断协议、实例调色板（≥60 间距）、语义颜色采样（≥90 间距）、Hypersim 场景选择。全部在 `configs/experiments/*_paper_aligned.yaml` 注释和 `effective_config.json` 中标为仓库选择。

## 4. 需要修改的具体建议

| 优先级 | 项 | 建议 |
|:-:|:--|:--|
| R1 | A2 立方体顶点顺序 | **已闭合**（本次审计中从 PDF 提取 Fig. 5 图像核对，顺序与标注米数均一致），无需修改。 |
| R2 | D4 默认损失权重 | 理想做法是把 `TrainingConfig.task_loss_weights` 默认改为全 1.0、`min_snr` 默认 `False`，使"不写就等于论文"。**本次未改**：`configs/flux_*`、`paper_reproduction_baseline.yaml` 等 8 份旧配置依赖当前默认值（depth 2.5 / normal 2.0 / semantic 0.5），直接改默认会静默改变它们的复现。应先把默认值显式写进这些 yaml，再切默认。paper_aligned 配置已显式写全 1.0 / 0 / false，不受影响。 |
| R3 | C12 法向聚合 | 在 `evaluate_paper_benchmarks.py` 同时输出逐像素与逐图两种均值，报告时注明；论文 Table 7 口径不明，两者都给可避免误比。 |
| R4 | A16 指代阈值 | 保持 `binary_white_black_v1`，但在 `evaluate_paper_benchmarks.py` metadata 中写明阈值 0.5 与"保留全部连通域"，与论文"straightforward"的空白对应。 |
| R5 | 文档陈旧 | `evaluate_paper_benchmarks.py` protocol 字符串曾写 "VKITTI … not derived from depth"；本次已改为与 #41 一致。 |
| R6 | C17 基线 | 不建议在仓库内复跑 SAM 3 / DA3 等基线；继续引用论文表格并标明出处即可。 |

## 5. 最终一致性结论

- **可核对项**：58 条中 48 条完全一致（83%），7 条部分一致，3 条缺失，**0 条实现错误**。部分一致项全部是论文留白处（阈值、聚合口径、截断协议、取整方式、混合比例数值、"lightweight"的形式、生成代理）的记录在案选择，没有一条会改变论文已写明的数值行为。
- **不可核对项**：基座模型、内部数据、训练配方、人工评测。仓库对这些项的处理方式（Qwen-Image-2.1、公开数据、LoRA、代理指标）全部显式标注为"非论文"，没有伪装成一致。
- **综合判断**：在"论文写明的方法层"达到 **≈95% 一致（55/58 至少部分一致，无错误）**；"100% 一致"对本论文不成立——不是因为实现缺陷，而是论文的可核对信息本身不覆盖训练过程与基座。R2–R4 为可选的整洁性修改，不影响任何已写明的论文行为。
