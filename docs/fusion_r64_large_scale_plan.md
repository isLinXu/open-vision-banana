# Fusion 固定基座的大规模 LoRA r64 训练方案（审阅稿）

关联 [Issue #108](https://github.com/isLinXu/open-vision-banana/issues/108)。

状态：**方案与数据接入实现；不启动训练，不合并 PR。完整主池尚未构建完成，缺失项见第2节。** 参考 [Issue #97](https://github.com/isLinXu/open-vision-banana/issues/97)，但本轮只有一条主线，不复制其初始化、rank、LR、分辨率消融矩阵。所有超参数是拟议的工程选择，不是已验证最优值。

## 1. 模型与主配方

- 唯一起点：[Melo111/Fusion-C1-KQ0r128-60-40](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40/tree/9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d)，固定 revision `9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d`。
- 完整 standalone QwenImage21Pipeline，已经融合 C1/K-Q0-r128 60/40。**不再加载或合并父 adapter，不把该模型当训练状态恢复**。
- 新建零初始增量 LoRA，r64、alpha64、rsLoRA=true；冻结 Transformer 原权重、text encoder、VAE。target modules 沿用 C1 的七个模块，见 YAML。缩放 alpha/sqrt(r)=8。
- 一个 seed42、512 分辨率的 SFT run，**100,000 次成功 optimizer update**，目标global batch32（按真实训练环境确定microbatch、world size与gradient accumulation）；共3.2M成功更新样本 draw，是 #97 20k 主线的5倍步数。不做多rank/LR/full/768搜索。
- BF16、无量化、gradient checkpointing；AdamW LR3e-5、betas(.9,.999)、eps1e-8、weight decay .01；cosine，warmup2000，clip1/drop10；EMA .9999/warmup500。
- 保留既有 flow、PNG target、resize/crop、prompt、task loss、零辅助loss；query_v2。不引入GRPO/OPD/新损失或训练时重编码。
- 固定配比 depth .35、semantic .20、instance .20、referring .15、normal不参与训练（0）、generation .10。使用现有显式task_sampling_weights，任务内沿用现有采样，不新建分源sampler。`max_samples_per_task: null`。

拟议 YAML：[fusion_r64_large_scale.yaml](../configs/experiments/fusion_r64_large_scale.yaml)。它表达目标配方，**不是当前main可直接运行的认证配置**，见第6节。

## 2. “最全数据”的真实边界

2026-11-08 核对main为 `a79aa0df54796fcdd375d84822e382bdb8ca444f`。#98/#99/#100/#103/#104/#106已合并。数据清理 [PR #107](https://github.com/isLinXu/open-vision-banana/pull/107) 仍 open/draft，头 `adb4a73a166a3212681641b943a77372115131a4`，不得假称已经在main生效。

以下是**组件发布数量的算术候选，正式接入后的行数以实际manifest为准**。遵循用户最新要求，本方案不安排数据去重，深度准入直接依赖#107，不另建一套清理流程：

| 任务 | 候选train行数 | 来源/限制 |
|---|---:|---|
| depth | 407,278 | 旧60,852 + LingBot50,000 + indoor292,153 − Hypersim1,066 + 新DIODE5,339 |
| normal（不启用） | 0 | 既有8,680行资产保留，不进入本轮训练 |
| semantic | 146,027 | 旧39,133 + COCONut106,894 |
| instance | 233,133 | 旧38,596 + COCONut194,537 |
| referring | 282,007 + N | 完整RES 270,195 + 既有Rea2Seg 11,812 + 全量LLMSeg实际准入N；替换旧32,932与4,304，不重复拼入 |
| generation | 16,492 | 既有replay |
| **启用任务合计** | **1,084,937 + N** | 原候选851,978 − 32,932 − 4,304 + 270,195 + N；不是已构建数量 |

这不是唯一图片数。本轮不设计或执行数据去重，不把不同任务的同图视为需要再次处理的工作项。旧depth60,852包含TartanAir38,661、原Hypersim8,924、增量Hypersim13,267。#107已处理增量Hypersim13,267→12,201（拒绝1,066）；所有训练深度的准入行为沿用该PR，不在本方案重复实现。407,278是按发布数量及已知替换得到的候选值，若实际#107接入后的数量不同，则更新配比曝光表，不绕过其检查。

**NYU不接入训练，包括已下载但未纳入的NYU labeled/raw；SUN仅non-NYU部分。** 不删除现存NYU资产。SNVC不可逆/单位未知深度、ReasonSeg train、未经审计的原始大池不算“已可用最全数据”。可能存在的既有195行DIODE额外组件必须先证明来源、train身份、native质量、未包含关系，再决定是否增加；本表不偷偷计入它。已有996行在indoor292,153中，只计一次。

### Referring 49,048 的计数口径

该值来自固定主池审计，不是完整RefCOCO-family库的大小：
**32,932旧referring + 11,812 Rea2Seg + 4,304 LLM-Seg = 49,048 train行**。同一主池另有2,296 val、2,322 test，均不计训练数；一行是image–expression/question–mask配对，不是唯一图像。

Rea2Seg在发布补充包中为15,275，历史512裁剪可见性过滤后主池为11,812；现存filtered manifest逐行重数为gqa-seg6,789、lisa-plus4,888、spatial135。LLM-Seg发布COCO子集5,000，过滤后主池4,304。旧32,932由原33,009再删除历史77条产生；现有固定审计未列它在RefCOCO/+/g三来源的逐项拆分，不能套用独立RES大库的比例冒充。

证据：[主池审计](https://huggingface.co/khazic/ovb-i88_t0_full_replay10_20261029_2202/blob/918a8a0b1a77e4542e996ae291f9882a2fc0721e/data_audit/pairs.audit.json)、[旧池审计](https://huggingface.co/khazic/ovb-i88_t0_full_replay10_20261029_2202/blob/918a8a0b1a77e4542e996ae291f9882a2fc0721e/data_audit/pairs.i88.audit.json)、[已发布RES完整库说明](res_data.md)。主池manifest SHA256为a7100a9ac0f43e72b647f7cd4f36d417175a44a683549ef98e083927f2e93806。历史49,048由现存补充manifest重数佐证；本次授权扩容使用下面的替换式入口。

### 已授权完整Referring接入

用户已确认完整LLMSeg包含其EgoObjects图片。目标为完整RefCOCO/RefCOCO+/RefCOCOg发布train + LLMSeg全训练问题（COCO和EgoObjects），保留现有Rea2Seg；任务比例和3.2M draw预算不变。

- 完整RES固定发布：`Melo111/open-vision-banana-res@dce183e8441c55c8fcf58afd87f90c776024c828`。使用`training.jsonl`的270,195 train行，分别为102,698 / 95,501 / 71,996；24,554安全val另存，不能计作训练。原官方321,327 train表达式已按既有发布协议排除held-out图像和确切重复，不重新做去重工作项。该文件SHA256为`786f30bb64bed54fd57b35fe5a719967fe738c9592c8b6a08003cf2215a6cf15`。
- LLMSeg依赖[PR #110](https://github.com/isLinXu/open-vision-banana/pull/110)的full-train入口：原始train 44,780问 = COCO 33,755 + EgoObjects 11,025，不再每图只取一问或封顶5,000。`N`必须来自该入口最终实际manifest，不可直接写44,780。既有面积规则初步通过31,270问也不是最终N，尚需实际图片、held-out及既有质量约束完成核验。官方val 3,905问不能进入train。
- 替换而非叠加：移除旧主池referring训练49,048行，再插入完整RES、最终LLMSeg和原11,812 Rea2Seg。旧32,932的逐来源身份目前未恢复，不声称逐条证明其是完整RES的子集；本方案明确用完整来源替代旧组件，而不是依据猜测做去重。旧验证/测试保留分区。
- 图像、表达式、掩码分别遵守原数据条款。LLMSeg标注和EgoObjects许可/访问步骤仍待核实；本PR不表示取得再分发许可，不上传新数据或改变上游条款。数据下载/本地接入不等于发布授权。

**当前实物状态**：本机完整RES旧目录仅剩metadata并带`LOCAL_PAYLOAD_REMOVED.md`，不能当完整恢复。新目录已校验固定发布metadata，首个payload下载遇到TLS握手失败，尚无有效完整恢复。其他深度/COCONut/LingBot组件在测评机存在，但真实训练服务器仍需自行绑定数据位置。当前Windows及测评机未找到原主池`data/qwen_main_depth_reasoning/pairs.jsonl`与资产；固定模型发布仅有`data_audit`统计，没有可恢复的主池manifest/资产。因此完整主池实体构建被原主池输入缺失阻塞，不能从统计虚构数据。LLMSeg输出尚待#110完成。


### 固定组件与接入方式

| 组件 | 固定版本 | 接入说明 |
|---|---|---|
| 旧联合池 | HF revision `918a8a0b1a77e4542e996ae291f9882a2fc0721e`；历史manifest SHA256 `a7100a9ac0f43e72b647f7cd4f36d417175a44a683549ef98e083927f2e93806` | 从现有冻结池恢复并核对来源；此处SHA只证明历史文件身份，不证明最新清理完成 |
| LingBot50k | depth repo `78946c7b0aff52b87fe2ad15143e93625a43b139` / lingbot-depth-v1 | 49个RobbySim场景，已发布；不是整个LingBot均匀样本 |
| COCONut | segmentation repo `b28483c48a57bb8338912da02d2e00c03d6da3d7` / coconut-b-v1 | 301,431 train、6,167 val；见[组件指南](coconut_segmentation.md) |
| indoor observed98 | depth repo payload `5a864bd118b5bca23e814917b62c0cb9f1f56da6` / indoor-observed98-v1 | 292,153 train、8,602 val；卡片版本b650…不替代payload版本；见[指南](indoor_depth_observed98.md) |
| clean Hypersim | #107固定头与排除清单 | 12,201 train，替换旧13,267，不能追加导致重复 |
| 新DIODE | depth repo `9ed5e65718db68a6cdd000754de681336ab18179` / diode-indoor-train-additions-v1 | 5,339 train；standalone diode_native_tar_v1，不是res_bundle |
| 完整RES | res repo `dce183e8441c55c8fcf58afd87f90c776024c828` | 270,195 train；替换旧32,932，安全val另存 |
| 全量LLMSeg | PR #110固定构建输入及产物SHA | COCO+EgoObjects全部训练问题，最终准入N待核验；替换旧4,304 |

depth repo是 `Melo111/open-vision-banana-depth`，segmentation repo是 `Melo111/open-vision-banana-segmentation`。旧池需从现有provenance补齐准确repo/subfolder及每个来源身份，不凭revision猜repo。正式lock未完成前不启动。

DIODE新组件官方train accounting：8,574=既有996+195排除 +428覆盖率拒绝 +1,616编码误差拒绝 +5,339保留；新组件与这些既有source ID互斥。已验证clean Hypersim+新DIODE **17,540 train、0排除、无NYU**，但它只是最小组合，不能替代完整五个启用任务池。服务器既有验证报告与COMPLETE绑定读取通过，本次没有重跑全资产像素审计。

### 数据接入约定（不重复清理工作）

1. 复用现有组件与manifest，记录固定revision和实际行数。LingBot、COCONut、indoor使用现有 `scripts/download_res_dataset.py`；新DIODE按[#107指南](https://github.com/isLinXu/open-vision-banana/blob/adb4a73a166a3212681641b943a77372115131a4/docs/clean_depth_data.md#published-diode-component)的pinned下载/校验/解包流程，不把diode_native_tar_v1当res_bundle。
2. 用#107输出的12,201行clean Hypersim替换旧13,267行，接入新DIODE5,339；不会再次执行自建深度准入/筛选方案，不改变数据处理、target或loss。
3. 使用严格组合入口 `python scripts/compose_fusion_training_pool.py --spec <已绑定的组件JSON> --output-dir data/fusion_r64_full_referring`。模板见 `configs/datasets/fusion_r64_pool_spec.example.json`；必须绑定真实root/manifest、SHA及LLMSeg最终N。`training.jsonl`只含train与baseline原val；原test及新增component held-out另存，不自动扩大dev或重新划分split。NYU不接入。
4. 组合器只在新目录重定位资产路径、记录来源SHA/替换账本/实际counts，拒绝缺资产、错数量、未知split或不完整替换，不改写共享资产。`COMPOSED_NOT_AUDITED`仅表示完成组合，不是训练ready；仍需#107 reader、RES/LLMSeg现有准入与真实训练服务器预检。原主池缺失时不会写成功标记。

## 3. 配比选择与曝光预算

**推荐唯一主线：depth35%、semantic20%、instance20%、referring15%、normal0%（不启用）、generation10%。** 不进行配比网格搜索或阶段性动态调整。理由：

- 继续保留10% generation replay，避免只增加vision数据而把保留能力目标丢掉。
- depth在旧候选池约47.8%；扩容后自然比例为407,278/(1,084,937+N)。继续固定35%，不因referring扩容自动改变已审阅的任务权重。
- semantic/instance各20%，覆盖COCONut新增大量数据；不按instance行数额外放大其任务权重，因同图单实例/多实例行并非独立图像。
- 按用户要求，将原normal的5%全部转给referring：referring15%加强查询/推理分割，normal不参与训练。其余任务比例不变，不声称这是验证得到的最优解。

| 任务 | 固定概率 | 100k×32期望draw | 候选池 | 平均draw/行 |
|---|---:|---:|---:|---:|
| depth | 35% | 1,120,000 | 407,278 | 2.75 |
| normal（不启用） | 0% | 0 | 0（原8,680不纳入） | 不适用 |
| semantic | 20% | 640,000 | 146,027 | 4.38 |
| instance | 20% | 640,000 | 233,133 | 2.75 |
| referring | 15% | 480,000 | 282,007 + N | 480,000 / (282,007 + N) |
| generation | 10% | 320,000 | 16,492 | 19.40 |

总预算仍为3.2M draw；相较本PR前版，normal的160k draw全部转给referring，使其从320k增至480k（+50%）。新池referring自然比例为(282,007+N)/(1,084,937+N)，generation为16,492/(1,084,937+N)；仍采用固定15%/10%。若所有44,780上游问题均通过，理论上界为326,787 referring、1,129,717总行、1.47 draw/行，但这不是实际准入结果。

现有Trainer在显式task_sampling_weights非空时不会再自动套vision_ratio：YAML仅列出五个启用任务的权重（总和1），并使用uniform基础策略；这与task_loss_weights不同，启用vision任务loss权重仍全部1，不双重加权。vision_ratio .9仅保留语义一致性，实际抽样以显式权重为准。现有sampler拒绝零权重，因此不写normal: 0：从data.tasks、task_sampling_weights及task_loss_weights中删除normal。Trainer仅为显式tasks建立dataset/group，故normal不会进入采样，即使输入manifest保留其历史行也不训练这些行；不修改sampler或删除资产。

任务内维持现有按行均匀采样。按发布行数估算depth内部：ARKit约71.3%、LingBot约12.3%、TartanAir约9.5%、clean/原Hypersim约5.2%、DIODE约1.6%、SUN约0.2%；对应整体draw大致24.96%、4.30%、3.32%、1.82%、.54%、.07%（舍入）。ARKit的相关帧/近距离倾向是主要分布风险；本版不添加分源sampler或重新筛选数据。

这些是期望draw，不是完整epoch或唯一图像数；有放回近似depth唯一覆盖约93.6%，不保证每行访问。记录实际各任务/source曝光及梯度丢弃。正常训练不为追逐“100%覆盖”临时改配比。小任务重复仍可能过拟合，使用固定dev曲线诊断；如后续需要改比例，作为用户审阅后的新版配方，而非暗中调参。

## 4. 保存、恢复与评测

- step0 + 每2,500步完整状态/online adapter，共40个训练节点；每2500执行内置flow-loss validation，并进入独立dev metric队列。内置eval_every不是图像推理指标。
- 主终点100k online；EMA单列，不用测试集或训练loss挑“best”。每10k阶段汇总固定dev/稳定性，最后做完整外部测试。不中途为了漂亮结果换主终点。
- dev仅来自训练来源已隔离的val/calibration：建议depth512、semantic256、instance256、referring256、generation64 prompts；数量不足就报告真实N，不重复补足。复用已隔离的既有划分，按scene/source分层；固定索引SHA。禁止NYU test/外部final split用于选LR、混合比例或checkpoint。
- 外部最终：NYU654（仅测试）、ETH3D454、DIODE Indoor325、Cityscapes500、RefCOCOg4896、ReasonSeg200（original query），generation固定200prompt proxy；normal不列为本轮训练/常规dev任务；instance若缺合规外部评测仅报dev，不冒称全能力验证。
- 固定BF16、512、50 steps、guidance4、strength.85、batch1、seed42+原始index，各benchmark自身mask/crop/cap。raw AbsRel/RMSE/delta1为主，aligned单列；DIODE极端decoded深度不得暗裁。DIODE引入训练后属于same-source held-out，不再声称zero-shot。历史Fusion test结果已被看到，比较应披露而非宣称盲测。
- 度量与队列记录run/step/online或EMA/protocol hash，完成checkpoint才入队，缺节点显示pending/failed不当0；rank分片合并原始累积量而非直接平均RMSE。每样本指标与失败记录保留。
- 训练与评测的调度由真实部署环境确定；checkpoint完成后进入评测队列，缺失节点明确标记，不假定训练与当前测评服务器共用硬件。
- full-state resume必须具备adapter、optimizer、scheduler、EMA、step、配置/数据/基座hash，缺任何必要项拒绝。首次新训不设resume；后续resume_weights_only=false，不能把父r128当r64恢复。
- 当前实现不保存完整RNG/data cursor，恢复会重新播种并从中断pass起点重新取数据，**非bit-exact续训**；记录新增启动与重复曝光。需要精确续训时先实现/test，而不是默许。
- 全部40个online节点与审计记录保留；完整状态至少最新两个、每10k里程碑与终点。该保留策略需后续实现，清理必须显式批准；不自动删除已有产物。

## 5. 与部署无关的训练预算

真实训练服务器不是当前测评服务器。本方案不以测评机的型号、卡数、显存、空闲状态或吞吐推断训练部署，不给出显存分析和基于测评机的工期估算。

目标预算保持100,000次成功更新、global batch32、合计3.2M成功draw。实际部署满足 global batch = world_size × per-process microbatch × gradient accumulation = 32；由真实训练环境确定这些分解参数。YAML现有microbatch1/accumulation4仅为待绑定的模板值，不声明真实world_size为8，也不能在任意进程数下直接声称global batch32。正式执行前按真实world_size修订并保存effective config；如改变global batch则须重新核对draw预算，不能静默变化。

本PR仍仅供审阅，不启动训练。后续在真实训练环境完成加载、前后向、保存及恢复验收；不在当前测评机代替执行或预留资源。
## 6. 当前代码阻断与启动验收

在main a79aa0d及#107观察到：
1. 已在本PR最小修复：TrainingConfig定义 `tune_mode` / YAML `training.tune_mode`，原validate与Trainer.full误访问 `train_mode`。现在校验和该属性统一读取已有 `tune_mode`，不新增字段或改变训练参数。默认、LoRA/full YAML及真实Fusion配置可加载；`--print-effective-config`验证不构造Trainer/模型。YAML `training.mode`、自造model_revision字段仍不支持。
2. Qwen2.1 backend明确拒绝4bit/8bit量化；LoRA仅data parallel，`--fsdp`/`--deepspeed`不支持，FSDP2是full参数专用。不要复制不存在的分片LoRA命令。
3. 旧run_lingbot_experiment.py要求父adapter/旧新配对，不适配当前fresh Fusion。需新的显式预检/启动入口。
4. #107未合并，完整五个启用任务入口接入与metric队列尚需完成。#107所报CI含既有失败，不是全绿。

实施顺序：
- [ ] 用户审阅主配方/预算。
- [ ] 固定#107合并后代码或经审阅的等价依赖；本PR已修复mode命名并测试配置加载/分支判定，实际fresh r64、resume与DDP smoke仍待执行。
- [ ] 接入全量组件与#107结果，记录真实counts；不新增去重/深度清理任务。
- [ ] 通过固定snapshot_download(revision=上述SHA)，核对provenance/file-manifest与所有权重shard，禁止浮动main；环境变量 `OPEN_VISION_BANANA_MODEL_PATH`指向已验本地快照，记录运行依赖版本。
- [ ] 预检有效配置、trainable参数、零初始adapter、基座hash；不再merge父权重。
- [ ] 完成独立dev metric队列、完整状态恢复检查、真实训练环境验收。
- [ ] 确认真实训练环境及执行授权，用户后续明确批准执行后才正式启动。计划PR合并本身不等于训练授权。

## 7. 本次验证范围

已核对GitHub状态、源码接口、数据发布文档/服务器既有17,540组合报告、固定模型发布CPU验证与真实shard字节、既有历史执行记录；本轮仅在隔离新目录准备数据和组合入口，未修改共享资产或运行任务。新增YAML已通过CPU语法、parser字段映射、rank/budget/global-batch/scheduler/EMA/resume契约检查；当前TrainingConfig动态load的命名缺陷已修复并通过真实加载回归，不伪称端到端训练/完整CI通过。模型训练、全量合并、训练环境验收、性能与最终指标均**未执行**。


CPU验收补充：在现有Python3.10环境中，更新后的实际TaskWeightedSampler五个任务概率与depth35/semantic20/instance20/referring15/generation10精确一致；8rank合计8000 draw不含任何normal索引，每rank1000；原main TrainingConfig.validate的train_mode AttributeError已动态复现，并在本PR修复。无GPU分配，无模型加载。

### 配置校验修复与剩余边界

调用链为 scripts/train.py -> load_config -> TrainingConfig.validate；原错误发生在Trainer创建之前，不是静默使用错误训练模式。除校验3处误引用外，仅更正Trainer.full的一处同名读取，避免通过校验后再次遇到同一AttributeError。去除重复mode校验，保留full模式禁止量化/weights-only恢复的规则，checkpoint metadata的train_mode键不改。

同一CPU/offline环境相关117项测试：原快照72失败/45通过，修复后8失败/109通过。8个新增字段回归全部通过，覆盖默认、LoRA/full YAML、真实Fusion YAML、错误mode、无Trainer的effective-config入口。更新既有full测试的过时字段/CLI拼写，保留原断言目的。

剩余8个full训练/保存/恢复测试揭露已有的full_finetune未导入NameError（trainer.py仍有调用而没有导入），与本次字段校验问题不同；未扩大修改，也不声称full训练端到端通过。LoRA主方案不走full分支，但实际训练环境验收仍未执行。仓库Ruff在相同环境下原/新均1587项，按路径、规则、消息和源码行比较无新增/消失；新回归文件单独check/format通过。已有hosted CI也包含缺少diffusers/cv2及其他collection问题，不将修复单个字段说成全仓CI已绿。

### 本轮完整Referring接入验证

- 新增严格组合CLI与10项合成数据CPU测试；本机Python 3.9执行10/10通过，Black格式化与`git diff --check`通过。仅以小型fixture验证契约，不冒充百万行实体池验证。
- 最新YAML语法、五任务显式权重、normal排除及r64断言通过；本轮只改变数据入口与注释，训练超参数不变。此前真实TrainingConfig/LoRA/full字段回归结论仍见上文；本轮没有重跑完整训练栈或全仓Ruff，不能沿用历史基线称当前全仓零新增问题。
- LLMSeg新增问题由#110按现有512裁剪可见性规则重新检查，不把旧4,304的排除列表当全量问题也已检查。完整RES沿用固定发布协议；没有把LLMSeg新发现的坏图自动跨来源扩散删除RES/Rea2Seg，所以270,195仍是组件数量而非全池联合验收承诺。真正训练前需完成既有reader/裁剪与split隔离验收，如实际准入数量变动须更新N和其他组件实数，不偷偷改变任务比例。
- 模板中的null SHA/N刻意让组合器失败，避免把未完成组件默认为可用。原主池及资产、完整RES恢复、LLMSeg两来源最终产物仍是实体组合前置条件。
