# Fusion 固定基座的大规模 LoRA r64 训练方案（审阅稿）

关联 [Issue #108](https://github.com/isLinXu/open-vision-banana/issues/108)。

状态：**仅方案与配置，不启动训练，不合并 PR。** 参考 [Issue #97](https://github.com/isLinXu/open-vision-banana/issues/97)，但本轮只有一条主线，不复制其初始化、rank、LR、分辨率消融矩阵。所有超参数是拟议的工程选择，不是已验证最优值。

## 1. 模型与主配方

- 唯一起点：[Melo111/Fusion-C1-KQ0r128-60-40](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40/tree/9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d)，固定 revision `9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d`。
- 完整 standalone QwenImage21Pipeline，已经融合 C1/K-Q0-r128 60/40。**不再加载或合并父 adapter，不把该模型当训练状态恢复**。
- 新建零初始增量 LoRA，r64、alpha64、rsLoRA=true；冻结 Transformer 原权重、text encoder、VAE。target modules 沿用 C1 的七个模块，见 YAML。缩放 alpha/sqrt(r)=8。
- 一个 seed42、512 分辨率的 SFT run，**100,000 次成功 optimizer update**，global batch32=8卡×1×4；共3.2M成功更新样本 draw，是 #97 20k 主线的5倍步数。不做多rank/LR/full/768搜索。
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
| referring | 49,048 | 既有RefCOCO族和已清理Rea2Seg/LLM-Seg，不再重复追加已包含的20,275 |
| generation | 16,492 | 既有replay |
| **启用任务合计** | **851,978** | 856,385 − 1,066 + 5,339 − 8,680，最终值以实际任务入口为准 |

这不是唯一图片数。本轮不设计或执行数据去重，不把不同任务的同图视为需要再次处理的工作项。旧depth60,852包含TartanAir38,661、原Hypersim8,924、增量Hypersim13,267。#107已处理增量Hypersim13,267→12,201（拒绝1,066）；所有训练深度的准入行为沿用该PR，不在本方案重复实现。407,278是按发布数量及已知替换得到的候选值，若实际#107接入后的数量不同，则更新配比曝光表，不绕过其检查。

**NYU不接入训练，包括已下载但未纳入的NYU labeled/raw；SUN仅non-NYU部分。** 不删除现存NYU资产。SNVC不可逆/单位未知深度、ReasonSeg train、未经审计的原始大池不算“已可用最全数据”。可能存在的既有195行DIODE额外组件必须先证明来源、train身份、native质量、未包含关系，再决定是否增加；本表不偷偷计入它。已有996行在indoor292,153中，只计一次。

### 固定组件与接入方式

| 组件 | 固定版本 | 接入说明 |
|---|---|---|
| 旧联合池 | HF revision `918a8a0b1a77e4542e996ae291f9882a2fc0721e`；历史manifest SHA256 `a7100a9ac0f43e72b647f7cd4f36d417175a44a683549ef98e083927f2e93806` | 从现有冻结池恢复并核对来源；此处SHA只证明历史文件身份，不证明最新清理完成 |
| LingBot50k | depth repo `78946c7b0aff52b87fe2ad15143e93625a43b139` / lingbot-depth-v1 | 49个RobbySim场景，已发布；不是整个LingBot均匀样本 |
| COCONut | segmentation repo `b28483c48a57bb8338912da02d2e00c03d6da3d7` / coconut-b-v1 | 301,431 train、6,167 val；见[组件指南](coconut_segmentation.md) |
| indoor observed98 | depth repo payload `5a864bd118b5bca23e814917b62c0cb9f1f56da6` / indoor-observed98-v1 | 292,153 train、8,602 val；卡片版本b650…不替代payload版本；见[指南](indoor_depth_observed98.md) |
| clean Hypersim | #107固定头与排除清单 | 12,201 train，替换旧13,267，不能追加导致重复 |
| 新DIODE | depth repo `9ed5e65718db68a6cdd000754de681336ab18179` / diode-indoor-train-additions-v1 | 5,339 train；standalone diode_native_tar_v1，不是res_bundle |

depth repo是 `Melo111/open-vision-banana-depth`，segmentation repo是 `Melo111/open-vision-banana-segmentation`。旧池需从现有provenance补齐准确repo/subfolder及每个来源身份，不凭revision猜repo。正式lock未完成前不启动。

DIODE新组件官方train accounting：8,574=既有996+195排除 +428覆盖率拒绝 +1,616编码误差拒绝 +5,339保留；新组件与这些既有source ID互斥。已验证clean Hypersim+新DIODE **17,540 train、0排除、无NYU**，但它只是最小组合，不能替代完整五个启用任务池。服务器既有验证报告与COMPLETE绑定读取通过，本次没有重跑全资产像素审计。

### 数据接入约定（不重复清理工作）

1. 复用现有组件与manifest，记录固定revision和实际行数。LingBot、COCONut、indoor使用现有 `scripts/download_res_dataset.py`；新DIODE按[#107指南](https://github.com/isLinXu/open-vision-banana/blob/adb4a73a166a3212681641b943a77372115131a4/docs/clean_depth_data.md#published-diode-component)的pinned下载/校验/解包流程，不把diode_native_tar_v1当res_bundle。
2. 用#107输出的12,201行clean Hypersim替换旧13,267行，接入新DIODE5,339；不会再次执行自建深度准入/筛选方案，不改变数据处理、target或loss。
3. 配置入口为 `data/fusion_r64_full_clean/pairs.jsonl`。按各组件原split接入；原validation独立保留，不拿17,540 train-only最小组合替代完整五个启用任务池。NYU不加入。
4. 在正式运行effective config中记录各来源manifest身份和实际task/source行数。使用#107已有reader检查，不禁用guard。此PR只提交接入说明，不构造/改写大数据资产或假造完成标记。

## 3. 配比选择与曝光预算

**推荐唯一主线：depth35%、semantic20%、instance20%、referring15%、normal0%（不启用）、generation10%。** 不进行配比网格搜索或阶段性动态调整。理由：

- 继续保留10% generation replay，避免只增加vision数据而把保留能力目标丢掉。
- depth约占启用候选数据47.8%，其中大量是相邻ARKit帧；给35%比原18%更充分，又不让相关视频帧按池规模完全主导训练。
- semantic/instance各20%，覆盖COCONut新增大量数据；不按instance行数额外放大其任务权重，因同图单实例/多实例行并非独立图像。
- 按用户要求，将原normal的5%全部转给referring：referring15%加强查询/推理分割，normal不参与训练。其余任务比例不变，不声称这是验证得到的最优解。

| 任务 | 固定概率 | 100k×32期望draw | 候选池 | 平均draw/行 |
|---|---:|---:|---:|---:|
| depth | 35% | 1,120,000 | 407,278 | 2.75 |
| normal（不启用） | 0% | 0 | 0（原8,680不纳入） | 不适用 |
| semantic | 20% | 640,000 | 146,027 | 4.38 |
| instance | 20% | 640,000 | 233,133 | 2.75 |
| referring | 15% | 480,000 | 49,048 | 9.79 |
| generation | 10% | 320,000 | 16,492 | 19.40 |

总预算仍为3.2M draw；相较本PR前版，normal的160k draw全部转给referring，使其从320k增至480k（+50%）。按851,978启用池纯比例抽样时referring约5.75%、generation约1.93%，不符合本轮强化指代与保留replay的目的，因此继续使用固定权重。

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
- 不额外抢占第9张GPU。8卡都用于训练时，可在完整checkpoint后协调暂停做评测，或明确排队延后；预估额外评测时间。不得与共享任务争抢或中断他人。
- full-state resume必须具备adapter、optimizer、scheduler、EMA、step、配置/数据/基座hash，缺任何必要项拒绝。首次新训不设resume；后续resume_weights_only=false，不能把父r128当r64恢复。
- 当前实现不保存完整RNG/data cursor，恢复会重新播种并从中断pass起点重新取数据，**非bit-exact续训**；记录新增启动与重复曝光。需要精确续训时先实现/test，而不是默许。
- 全部40个online节点与审计记录保留；完整状态至少最新两个、每10k里程碑与终点。该保留策略需后续实现，清理必须显式批准；不自动删除已有产物。

## 5. 资源与预算（估算，不是实测训练）

本次只读资源快照：服务器8×A100-PCIE-40GB，每卡40960MiB，当时均1MiB/0% GPU；主机约754GiB RAM、/data剩余1.7TB。未来启动前重新查GPU/进程/锁，当前空闲不代表预留授权。

固定Fusion发布验证列出的BF16文件：Transformer约14.230GB、text encoder约17.534GB、VAE约1.351GB，总约33.116GB（30.84GiB）。这不是通常假设的20B Transformer；按实际文件估算。8卡LoRA是**每卡复制**，不能用320GB总量证明单卡能装。40GiB减权重仅约9.2GiB，尚需activation、LoRA/Adam/EMA、attention workspace、allocator峰值。是否fit必须实测，不能承诺。

审阅后先单卡forward/backward/save/reload，再8卡200-step资源smoke（不计正式预算）。保留BF16而非未经支持的量化；测峰值reserved/allocated、宿主内存、每成功update耗时、checkpoint大小/写时、恢复一致性。若OOM，先实现并验证冻结encoder offload/适当sharded LoRA支持或改硬件；**不擅自降rank、换基座、改global batch或量化**。这不是增加消融，而是可运行性验收。

以8卡100k更新举例：
| 实测update秒数假设 | 纯训练天数 | GPU-hours |
|---:|---:|---:|
| 5 | 5.79 | 1,111 |
| 10 | 11.57 | 2,222 |
| 20 | 23.15 | 4,444 |
未计smoke、保存、dev/外部评测、故障与I/O；用200步稳定段的p50/p95替换，不将表格当性能承诺。不得误用microstep耗时当optimizer update耗时。

磁盘公式：现有共享只读资产 + 新manifest/必要解包 + 基座33.1GB + 40×adapter大小 + 保留完整状态数×state大小 + predictions/logs +至少20%余量；每个量用真实文件统计。不复制整个原始库或下载86.75GB DIODE源archive。小模型权重并不等于checkpoint仅LoRA大小，optimizer与EMA单独计量。

## 6. 当前代码阻断与启动验收

在main a79aa0d及#107观察到：
1. TrainingConfig定义 `tune_mode` / YAML `training.tune_mode`，validate与Trainer.full访问 `train_mode`，产生AttributeError。需一致修复并回归。YAML `training.mode`、自造model_revision字段都不支持。
2. Qwen2.1 backend明确拒绝4bit/8bit量化；LoRA仅data parallel，`--fsdp`/`--deepspeed`不支持，FSDP2是full参数专用。不要复制不存在的分片LoRA命令。
3. 旧run_lingbot_experiment.py要求父adapter/旧新配对，不适配当前fresh Fusion。需新的显式预检/启动入口。
4. #107未合并，完整五个启用任务入口接入与metric队列尚需完成。#107所报CI含既有失败，不是全绿。

实施顺序：
- [ ] 用户审阅主配方/预算。
- [ ] 固定#107合并后代码或经审阅的等价依赖；修复mode命名，覆盖config load、fresh r64、resume、DDP测试。
- [ ] 接入全量组件与#107结果，记录真实counts；不新增去重/深度清理任务。
- [ ] 通过固定snapshot_download(revision=上述SHA)，核对provenance/file-manifest与所有权重shard，禁止浮动main；环境变量 `OPEN_VISION_BANANA_MODEL_PATH`指向已验本地快照，记录运行依赖版本。
- [ ] 预检有效配置、trainable参数、零初始adapter、基座hash；不再merge父权重。
- [ ] 完成独立dev metric队列、完整状态恢复检查、资源smoke与必要内存支持。
- [ ] 重新检查GPU所有权，用户后续明确批准执行后才正式启动。计划PR合并本身不等于训练授权。

## 7. 本次验证范围

只读核对GitHub状态、源码接口、数据发布文档/服务器既有17,540组合报告、固定模型发布CPU验证与真实shard字节、当前GPU/CPU/磁盘；未修改共享数据、训练代码或运行任务。新增YAML已通过CPU语法、parser字段映射、rank/budget/global-batch/scheduler/EMA/resume契约检查；当前TrainingConfig动态load的既有命名缺陷应作为blocked记录，不伪称端到端训练/完整CI通过。模型训练、全量合并、GPU smoke、性能与最终指标均**未执行**。


CPU验收补充：在现有Python3.10环境中，更新后的实际TaskWeightedSampler五个任务概率与depth35/semantic20/instance20/referring15/generation10精确一致；8rank合计8000 draw不含任何normal索引，每rank1000；当前main TrainingConfig.validate的train_mode AttributeError已动态复现。无GPU分配，无模型加载。
