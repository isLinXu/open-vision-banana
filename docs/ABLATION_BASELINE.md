# 第一份共享基线

一个共享 LoRA，评测只换 prompt。这是对齐 Vision Banana 的第一份基线，不是 Table 1。下面每一行是一份单独的 LoRA。

| 对照 | 配方 | 指标 | 值 |
|---|---|---|---|
| SANA CS-19，15k 中 step 8000 | lr 5e-5，min-SNR 开，flow shift 1，cfg dropout 0.1，batch 1 × accum 8，cosine | Cityscapes val mIoU | 0.266 |
| SANA 指代 LoRA | 与 CS-19 不同 | RefCOCOg UMD val cIoU | 0.143 |
| FLUX 语义 LoRA | 语义 specialist，不是指代 | RefCOCOg UMD val cIoU | 0.098 |
| SANA 法向 LoRA | 单任务 | DIODE-Indoor mean° | 24.96 |
| SANA、FLUX 深度 | 单任务或旧配方 | 度量深度闸门 | 失败 |

主表四项：Cityscapes val mIoU、RefCOCOg UMD val cIoU、NYU δ1 与 AbsRel、DIODE-Indoor 法向 mean°。入表 checkpoint 按这四项选。一行结果同时要有基座、配比、步数、末步学习率。缺一项就不能下结论。`cityscapes8`、ADE-150 交叉色板、匈牙利重排和失败 preflight 不进主表。训练不使用 NYU `official_test`。本仓库数字不写成 Table 1。

和 specialist 的差距同时包含「共享一个 LoRA」和「配方不同」。不能把差距写成只有前者。

## 固定条件

这是本轮的控制配方，不是已经测过的最优组合。下一节的数字来自时间步修复前的单因素比较；`best3` 在那份报告里尚未跑完。这里把单因素里超过 seed 波动的改动叠在一起，只为了让配比、步数、基座可比。

SANA-1600M，原生时间步，分辨率 512，`logit_normal_timesteps: true`。LoRA rank 32 / alpha 64，`rs_lora: true`。`flow.min_snr: false`，`flow.shift: 3.0`，`flow.cfg_dropout_rate: 0`，`training.lr: 1.0e-4`。`scheduler.type: constant`，`warmup_ratio: 0.05`。`cosine_restarts` 的周期不是配置字段，默认是总步数的三分之一，本轮不用。`training.batch_size` 乘 `gradient_accumulation` 等于 32。`loss.task_loss_weights` 里 depth、normal、semantic、referring 都是 1.0。`normal_angular_loss_weight: 0`，`depth_edge_aux_weight: 0`。训练器默认的损失权重是 depth 2.5、normal 2.0、semantic 0.5，不写明就会和采样配比混在一起。`vision_ratio: 1.0`。`depth_curve: metric_power`。`checkpoint.save_every: 1000`。

`flow.shift` 只作用于 SANA 的时间步采样。写到其他基座的配置里不会改变它们的时间步。

一次只改下表中的一个因素。配比要在 Cityscapes mIoU、NYU δ1、DIODE mean° 里至少两项优于 `uniform`，且没有一项退步，才替换 `uniform`。三项方向不一致时，胜出保持 `uniform`，备注写明哪一项退步。内部 held-out 的 AbsRel ±0.023 与 mIoU ±0.006 只说明那次网格的 seed 波动，不用来裁决主表。

| 因素 | 水平 | 要得出的结论 |
|---|---|---|
| 配比 | 四个任务：`uniform`；`sqrt`；semantic 权重 3、其余 1；depth 权重 3、其余 1。来源比例在这四组里不变 | 哪个任务被挤掉。DIODE mean° 单独写。内部网格里 normal 单任务 22.0°、三任务 26.2°、normal+seg 21.9°，冲突在 depth 与 normal，但是内部角度，不是 DIODE。Cityscapes 稀有类若四组都接近 0，下一步是类均衡。来源比例另见数据一节，不在这四组里改 |
| 步数 | 只用胜出配比：8k、30k；60k 仅当 30k 的主表仍高于 8k，且学习率仍是 1e-4 | 共享配方在步数上是否还在涨。和 specialist 比时注明配方不同 |
| 基座 | SANA 为控制。Qwen-Image-2.1 先注册并完成 200 step，源图进入视觉 token。FLUX.1-dev 4-bit 仅在 CLIP 77 token 保留颜色名和指代对象时加入 | 基座差别。非 SANA 不把 `flow.shift` 当成有效因素 |

不进这张表：FLUX Kontext、HunyuanImage-3、Joy、LongCat。实例分割不进第一份训练。SA-Co/Gold 只作评测。

## 数据与来源比例

训练集是 [SenseNova-Vision-Corpus-50M](https://huggingface.co/datasets/sensenova/SenseNova-Vision-Corpus-50M)，再用 [Rea2Seg-16K](https://huggingface.co/datasets/snowball521/Rea2Seg-16K/tree/1b2247ca67e50e76f6494065ae089be4bd6ccd19) 补充指代。采样器只有任务这一层，没有数据集这一层。来源比例必须先写进 manifest，T1–T4 只改任务权重。

SNVC 公布的帧数是：结构化理解 18.9M，分割 1.3M，稠密几何 17.3M，多视图几何 12.5M。结构化理解是点、框和 OCR 的文本输出，多视图是重建。这两类不进本轮 manifest，否则按行数抽样时它们会盖过深度、法向和分割。稠密几何只拆成 depth 与 normal。分割只拆成 genseg（semantic）和 gcg（referring）。genseg 与 gcg 的划分要以 jsonl 实计为准，数据集卡没有给出这个比例。

Rea2Seg-16K 的 train 有 16090 行，全部是指代掩码：gqa-seg 7998，lisa-plus 7052，spatial 801，reasonseg 239。相对 17.3M 几何帧，16090 行大约是 0.09%。四个任务按原始行数做 proportional 时，Rea2Seg 几乎抽不到。指代池因此先按来源 1:1 重采样：Rea2Seg 全部 16090 行保留，SNVC gcg 用 seed 42 抽到同样 16090 行。depth、normal、genseg 保持全量。

按这个 manifest，uniform 给四个任务各 25% 的步。指代的一半来自 Rea2Seg，所以 Rea2Seg 占全部步数的 12.5%。其中 reasonseg 只有 239/16090，大约是全部步数的 0.19%。30k 步、有效 batch 32 时，reasonseg 大约出现一千八百次，同一行会重复。T1–T4 保持这个类别比例。把它改成四个类别等概率，是 T13，不和任务配比一起改。

语义行保留各自 jsonl 里的颜色图例。Cityscapes 的 genseg 用官方 19 色图例。行内没有 prompt 时，训练器会填上 “150-class … fixed training palette”，这种行不能进 manifest。Cityscapes val 仍用 `SEMANTIC_SEGMENTATION_PROMPT` 评测。semantic 加重提高的是全部 genseg，不只是 Cityscapes。T0 要写下 Cityscapes genseg 行数占全部 genseg 的比例。

Rea2Seg 的训练 prompt 用 `question`，目标是白掩码、黑背景。`cot_answer` 不写进控制组的 prompt。

SNVC 的图像目标在 `SenseNova-Vision-Corpus-50M.tar.gz000` 起的 406 个分卷里，每卷 20GB，合成一条 gzip。只有第一卷以 gzip 魔数开头。已查看的 80MB 前缀里只有 `51WORLD/gcg`。不能靠单独下载后面的分卷抽出 COCO 深度或 Cityscapes 掩码。Rea2Seg 的图和掩码在 parquet 里，可以单独落地。

## 已有内部消融（2026-10-02）

来源是 W&B 报告 [实验进度总结](https://forge.coreweave.com/wandb/khazzz1c/open-vision-banana/reports/--VmlldzoxODA0Mjc1Ng==)。57 个已评训练，口径是各 run 自己的 held-out val（depth 319、normal 276、seg 241），512px，EMA，20 步 Euler。主表只收官方划分上的新评测，并且加载保存下来的 online LoRA。`eval_cityscapes19.py` 读的是 `lora_path`，不会自动使用 `ema_weights.pt`。报告里的 mIoU 不是 Cityscapes mIoU，也不是 RefCOCOg cIoU。

当时的 SANA run 把时间步传成 σ∈[0,1]，原生约定是 σ×1000。同一约定内部的高低比较仍然有效。1200 步 depth 单变量对照：旧代码 AbsRel 1.65，修复后 1.19。新基线用当前训练器的原生时间步。

三 seed 基线（step 30000，bs4 × ga8，lr 5e-5，rank 32，min-SNR γ=5，cfg dropout 0.1）：AbsRel 0.712±0.023，角度 26.2±0.3°，mIoU 0.173±0.006。小于这个幅度的单因素改动，在那次网格里视为无效。

单因素结果如下。控制配方把其中的正向改动叠在一起，该组合没有在这份报告里测过。

| 改动 | step 30000 | 相对三 seed 均值 |
|---|---|---|
| 关掉 min-SNR | AbsRel 0.520，角度 21.0°，mIoU 0.209 | 三任务同时最好。γ=1 同方向但更弱 |
| flow shift 3 | 0.614 / 22.7° / 0.197 | 三项都更好 |
| lr 1e-4 | 0.629 / 24.0° / 0.191 | 三项都更好。lr 2.5e-5 为 0.960 / 28.7° / 0.145 |
| 关掉 rsLoRA | 1.011 / 30.7° / 0.140 | 全场最差，rsLoRA 保持开启 |
| rank 16 | 0.859 / 28.6° / 0.155 | 变差。rank 64 为 0.658 / 23.9° / 0.186，增益小于关掉 min-SNR，本轮仍用 32 |
| cfg dropout 0.2 | 0.811 / 28.8° / 0.169 | 变差。dropout 0 为 0.685 / 22.4° / 0.172 |
| 有效 batch 8 | 0.773 / 26.3° / 0.153 | 同样 30000 步约只看到 1/4 样本 |

任务组合，同一口径：depth 单任务 AbsRel 0.729，三任务 0.712。normal 单任务 22.0°，三任务 26.2°，normal+seg 21.9°。seg 单任务 mIoU 0.197，三任务 0.173。分割加 GCG 后 mIoU 0.198，AbsRel 退到 0.791。warmup 1% 与 EMA 0.999 落在 seed 波动内。

val loss 最低的 checkpoint 多在 24000–27500，指标与最终权重接近。`main_s1` 的 val 最低在 14000，评测差于 step 30000。关掉 min-SNR 后 val loss 约 0.64，与开启时约 0.20 不能横向比较。

FLUX.1-dev 为 4-bit LoRA。第一轮训练参考系有误。第二轮 5000 步（约 2 万样本；SANA 网格约 96 万样本）里，关掉 min-SNR 为 AbsRel 1.479、角度 27.1°、mIoU 0.143，是唯一 normal 低于 30° 的三任务 run；lr 1e-4 的 AbsRel 1.402，角度仍为 52.5°。单任务 normal 20.7°，小于 30° 的比例 0.783。rank、cfg dropout、分割加 GCG 在 5000 步内分不出差别。

生成代理用 COCO val2017 的 500 条 prompt，同 prompt 同 seed，对原模型比较 FID 与 LPIPS。FLUX 两轮 LoRA 的 FID 与原模型同为约 98.7。SANA `main_s42` 的 FID 从 105 到 165，但这份 LoRA 与生成管线的时间步不一致。共享基线的生成代理要等原生时间步的 LoRA 再测。

## 执行

从上到下取第一张 `状态: todo`、且依赖均为 `done` 或 `skip` 的卡。只做这一张。完成后改状态，并在文末结果表追加一行。不把两张卡并进同一次训练。状态只用 `todo`、`doing`、`done`、`blocked`、`skip`。

T6 在 T0 完成后就可以做，不排在 T1–T5 后面。T7 仍等 T5 和 T6。T9–T13 不阻塞 T1–T8，也不改已填的主表四项。

中途评测与入表评测分开。每 5000 步只跑 Cityscapes val 全量，以及 NYU、DIODE 的固定子集（同一 seed、同一下标，写进 JSON）。RefCOCOg 只在写入结果表的 checkpoint 上跑 UMD val 全量。SANA 的 Cityscapes 评测沿用 `scripts/eval_cityscapes19.py`：512，20 步，strength 0.85，seed 42，fp32，加载 online LoRA。

### T0 锁定配置

状态: done  
依赖: 无

新建 `configs/experiments/sana_1600m_shared_baseline.yaml`，不改 `configs/experiments/sana_1600m_cityscapes19.yaml`。字段用上面的控制配方。manifest 按数据一节建成后再写配置。`data.tasks` 为 depth、normal、semantic、referring。`data.task_sampling: uniform`。配置注释写下四个任务的行数（depth 5630、normal 4575、semantic 4460、referring 32136，去掉 1315 行与评测集重图的行）、gcg 与 Rea2Seg 各 16068、Cityscapes genseg 占全部 genseg 的比例（311/4460 = 6.97%），以及四个 Rea2Seg 类别的行数。`data.task_weights: sqrt` 会把采样策略设成字符串 `sqrt`，不是数值权重；数值权重只写在 `task_sampling_weights`。不要抄 `configs/paper_reproduction_baseline.yaml` 的 `vision_ratio: 0.1`。不要开始训练。

完成：`scripts/train.py --config <该文件> --print-effective-config` 可打印。打印结果里 `vision_ratio` 为 1.0，`min_snr` 为 false，`flow_shift` 为 3.0，`lr` 为 1e-4，`scheduler` 为 constant，`grad_accum` 为 8（`batch_size` 4 × `grad_accum` 8 = 32），四个任务的 `task_loss_weights` 都是 1.0。manifest 中没有 “150-class”，也没有结构化理解或多视图重建的行。

### T1 均匀配比

状态: done  
依赖: T0

复制 T0。`task_sampling: uniform`，`steps: 30000`，`task_sampling_weights` 为空。每 5000 步做中途评测，并记录学习率。学习率应保持 1e-4。不改损失权重、min-SNR、flow shift 或 LoRA rank。

完成：日志到 30000/30000。结果表有 `uniform` 一行，四项全量指标和末步学习率齐全。中途子集分数写在备注，不代替全量。

产物: `output/sana_1600m_shared_uniform/`（HF: `khazic/ovb-sana_1600m_shared_uniform`）

### T2 sqrt 采样

状态: done  
依赖: T1

复制 T1，只把 `data.task_sampling` 改为 `sqrt`。权重保持为空。`sqrt` 用的是重采样之后的行数：depth、normal、genseg 仍是全量，referring 只有 32136 行（referring 0.46，其余约 0.17–0.19）。因此 sqrt 仍然偏向几何任务。不改来源比例、步数和学习率。

完成：结果表新增 `sqrt` 一行。备注写相对 T1 哪个主表指标变好、哪个变差。

产物: `output/sana_1600m_shared_sqrt/`（HF: `khazic/ovb-sana_1600m_shared_sqrt`）

### T3 语义加重

状态: done  
依赖: T1

复制 T1，保持 `task_sampling: uniform`。只设 `task_sampling_weights`：semantic 3，depth 1，normal 1，referring 1。uniform 下概率为 1/2、1/6、1/6、1/6。不改损失，也不改 genseg 内部的数据集组成。

完成：结果表新增一行，Cityscapes mIoU 0.017（五组最高但仍接近 0）。motorcycle、truck、traffic light 仍接近 0。

产物: `output/sana_1600m_shared_semantic_heavy/`（HF: `khazic/ovb-sana_1600m_shared_semantic_heavy`）

### T4 深度加重

状态: done  
依赖: T1

复制 T1，保持 `task_sampling: uniform`。只设 `task_sampling_weights`：depth 3，normal 1，semantic 1，referring 1。uniform 下概率为 1/2、1/6、1/6、1/6。深度曲线保持 `metric_power`。不改稠密几何内部的数据源组成。

完成：结果表新增一行，含 NYU δ1 (0.066) 与 AbsRel (1.98)。`scripts/check_depth_gate.py` 对 NYU 平凡基线给出 fail。

产物: `output/sana_1600m_shared_depth_heavy/`（HF: `khazic/ovb-sana_1600m_shared_depth_heavy`）

### T5 步数

状态: done  
依赖: T2、T3、T4

按固定条件里的规则从 T1–T4 选出胜出配比，理由写入结果表。三项方向不一致时胜出就是 T1。T1 的 30k 保留为对照。`save_every: 1000` 时直接评 `checkpoint-8000` 的 online LoRA；30k 已存在则不重训。仅当 30k 的全量主表仍高于 8k，且该步学习率仍是 1e-4 时，才训到 60000。否则 60k 标 `skip`。不加步数的同时不改采样权重。

完成：自动选出 `sqrt`（在 Cityscapes mIoU 和 NYU δ1 上比 uniform 高 0.003 和 0.001，但差异落在噪声内），比较 8k 与 30k 后跳过 60k（`skip`）。

产物: 胜出目录下的 `checkpoint-8000`、`checkpoint-30000`（60000 为 `skip`）

### T6 Qwen-Image-2.1 冒烟

状态: done  
依赖: T0

在 `src/open_vision_banana/model_registry.py` 注册 `qwen-image-2.1`，pipeline 为 `QwenImage21Pipeline`，权重 `Qwen/Qwen-Image-2.1`。训练前向须把源图送入视觉 token，不沿用只编码文本的 Qwen / Joy `encode_text`。运行环境需提供该 pipeline，且 `transformers>=5.17`；缺依赖时使用单独环境，不改当前训练环境的 torch。先跑 200 step，通过前不跑完整步数。不纳入 HunyuanImage-3、Joy、LongCat。`flow.shift` 不会改变这个基座的时间步。

完成：配置可解析该基座；200 step 损失为有限值；源图进入视觉 token。

产物: `khazic/ovb-qwen_image_21_shared_smoke` (`checkpoint-200_final`)

### T7 Qwen-Image-2.1 对比

状态: doing  
依赖: T5、T6

使用 T5 的胜出配比、步数和学习率日程，只更换基座为 `qwen-image-2.1`，评同一主表。若必须量化，单独加一行，不与未量化的 SANA 行并成同一条件。

进展：`uniform` 已训满 30k（`khazic/ovb-qwen_image_21_shared_uniform/checkpoint-30000_final`），目前仅评了 5k 中途（Cityscapes mIoU 0.235，NYU 子集 δ1 0.130 / AbsRel 1.37），30k 全量评测待补；`sqrt` 重跑到约 6800 步时中断，最后保存的 ckpt 为 `checkpoint-6000`（5k 中途 Cityscapes mIoU 0.237，NYU 子集 δ1 0.147 / AbsRel 1.45）。

产物: `output/qwen_image_21_shared_baseline/`（HF: `khazic/ovb-qwen_image_21_shared_uniform`、`khazic/ovb-qwen_image_21_shared_sqrt`）

### T8 FLUX.1-dev

状态: blocked  
依赖: T5

记录 Cityscapes 19 色 prompt 与一条 RefCOCOg 句子在 CLIP 77 token 处截断前后的 token 数。颜色名或指代对象被切掉时，结果表记 `blocked`，不开训。否则用 T5 胜出配方跑 4-bit FLUX.1-dev，并保持 `min_snr: false` 与修复后的训练参考系。`flow.shift` 不改变 FLUX 的时间步。内部 held-out 的 5000 步 mIoU，以及已有语义 LoRA 的 cIoU 0.098，都不是这行的分数。

完成：CLIP 77 token 截断后，Cityscapes 19 个类名只剩 4 个，按卡片规则记 `blocked`。

产物: 截断记录（`blocked`）

### T9 nuScenes

状态: done  
依赖: 无

把 nuScenes val 前视相机的 LiDAR 投影为米制 `depth/*.npy`（与 `samples/CAM_FRONT` 同名），再按 80 m、无 crop 评 δ1 与 AbsRel。mini split 不计入。不改已填的主表四项。

完成：SANA `sqrt` 30k 在 6019 帧上测得 δ1 = 0.055，AbsRel = 0.631。

### T10 ReasonSeg

状态: done  
依赖: 无

用胜出 checkpoint 的 online LoRA 评 ReasonSeg val。报告 gIoU 时注明「原始查询，无 Gemini 改写」。不与论文 gIoU 0.793 写成同一协议。

完成：200 条样本，gIoU：uniform 0.209、sqrt 0.186、T13 0.193（原始查询，无 Gemini 改写）。

产物: `output/eval/reasonseg_val_<run>.json`

### T11 SA-Co/Gold

状态: blocked  
依赖: 无

核对 `queries.jsonl` 的行数与可解析图像数。出分时注明「annotator A，无 Gemini 存在性过滤」。不与论文 cgF1 47.5 对照。

完成：`blocked`，官方数据集 `facebook/SACo-Gold` 需要 HF 授权，当前账号尚无权限。

### T12 生成

状态: done  
依赖: 无

GenAI-Bench 1600 条 prompt、ImgEdit 767 条指令保持「未做人类评」。已有代理是 COCO val2017 的 500 条 prompt，同 prompt 同 seed，对原模型比较 FID 与 LPIPS。FLUX 的 FID 约 98.7，与原模型一致。SANA `main_s42` 的 FID 从 105 到 165，该比较混有时间步不一致。新的生成代理只对原生时间步的共享 LoRA 重测，列名写 proxy。论文胜率 53.5% 与 47.8% 不写入我们的列。

完成：SANA `sqrt` 30k proxy 测得 FID 388（基座约 105），LPIPS 0.965，无生成混合（`vision_ratio: 1.0`）下生成能力明显受损。

### T13 Rea2Seg 类别比例

状态: done  
依赖: T1

只改指代池里 Rea2Seg 四个类别的抽样，任务权重和 gcg:Rea2Seg = 1:1 保持不变。控制组按原始行数，reasonseg 占 Rea2Seg 的 232/16068。这一张改成四个类别等概率，spatial 与 reasonseg 需要重复采样。用 T1 的任务配比和步数。

完成：结果表新增一行。RefCOCOg cIoU 0.231（T1 uniform 为 0.229），ReasonSeg gIoU 0.193（T1 uniform 为 0.209，原始查询，无 Gemini 改写）。

产物: `output/sana_1600m_shared_rea2seg_balanced/`（HF: `khazic/ovb-sana_1600m_shared_rea2seg_balanced`）

## 结果表

训练卡在四项主指标和末步学习率齐备前，状态保持 `doing`。specialist 按任务分行，空着的格子表示那份 LoRA 没有这项评测。DIODE 的法向标注在本轮评测机缺失，因此主表 DIODE mean° 暂缺，补列 NYU / DIODE 对齐尺度 δ1 与 ReasonSeg gIoU 供诊断。

| Run | 状态 | 基座 | 配比 | 步数 | 末步学习率 | Cityscapes mIoU | RefCOCOg cIoU | NYU δ1 / AbsRel | NYU δ1 (对齐) | DIODE δ1 (对齐) | DIODE mean° | ReasonSeg gIoU | 共享 LoRA | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CS-19 specialist | done | SANA-1600M | 仅 semantic | 8000（15k 日程中的最好点） | 15k 末步为 5e-7；8000 步尚未贴底 | 0.266 | | | | | | | 否 | min-SNR 开，shift 1，cfg 0.1，accum 8，cosine |
| SANA 指代 | done | SANA-1600M | 仅 referring | | | | 0.143 | | | | | | 否 | 不是本控制配方 |
| FLUX 语义 | done | FLUX.1-dev | 仅 semantic | 15000 | | | 0.098 | | | | | | 否 | 语义 LoRA 上的 cIoU，不是指代 specialist |
| SANA 法向 | done | SANA-1600M | 仅 normal | | | | | | | | 24.96 | | 否 | |
| SANA / FLUX 深度 | done | 见实验记录 | 仅 depth | 30000 | | | | 0.066 / 1.990 | 0.427 | 0.322 | | | 否 | 度量闸门失败（训练目标为相对灰度） |
| T1 SANA uniform | done | SANA-1600M | uniform | 30000 | 1e-4 | 0.009–0.014 | 0.229 | 0.066 / 2.05 | 0.437 | 0.371 | 缺标注 | 0.209 | 是 | HF: `khazic/ovb-sana_1600m_shared_uniform` |
| T2 SANA sqrt | done | SANA-1600M | sqrt | 30000 | 1e-4 | 0.012–0.015 | 0.220 | 0.067 / 1.99 | 0.453 | 0.345 | 缺标注 | 0.186 | 是 | T5 自动选出（差异在噪声内）；nuScenes δ1 0.055 / AbsRel 0.631；生成 proxy FID 388 / LPIPS 0.965 |
| T3 SANA semantic 加重 | done | SANA-1600M | semantic 3:1:1:1 | 30000 | 1e-4 | 0.017 | 0.225 | 0.065 / 2.02 | 0.458 | 0.312 | 缺标注 | | 是 | Cityscapes 仍接近 0；HF: `khazic/ovb-sana_1600m_shared_semantic_heavy` |
| T4 SANA depth 加重 | done | SANA-1600M | depth 3:1:1:1 | 30000 | 1e-4 | ~0.01 | 0.163 | 0.066 / 1.98 | 0.452 | 0.330 | 缺标注 | | 是 | 深度闸门失败；HF: `khazic/ovb-sana_1600m_shared_depth_heavy` |
| T5 SANA sqrt @ 8k | done | SANA-1600M | sqrt | 8000 | 1e-4 | ~0.01 | 0.227 | 0.051 / 2.26 | 0.453 | 0.343 | 缺标注 | | 是 | 对比 30k 后 60k 记 `skip` |
| T13 SANA Rea2Seg 等概率 | done | SANA-1600M | uniform (Rea2Seg 4类等权) | 30000 | 1e-4 | ~0.01 | 0.231 | 0.069 / 1.94 | 0.373 | 0.318 | 缺标注 | 0.193 | 是 | HF: `khazic/ovb-sana_1600m_shared_rea2seg_balanced` |
| T7 Qwen-Image-2.1 uniform | doing | Qwen-Image-2.1 | uniform | 5000（已训满 30k，待评 30k） | 1e-4 | **0.235** | 待跑 | 0.130 / 1.37 (100张子集) | | | 缺标注 | | 是 | 5k 步已学会 Cityscapes 语义分割；HF: `khazic/ovb-qwen_image_21_shared_uniform` |
| T7 Qwen-Image-2.1 sqrt | doing | Qwen-Image-2.1 | sqrt | 5000（6800步中断，存至6000） | 1e-4 | **0.237** | 待跑 | 0.147 / 1.45 (100张子集) | | | 缺标注 | | 是 | HF: `khazic/ovb-qwen_image_21_shared_sqrt` |
| T8 FLUX.1-dev | blocked | FLUX.1-dev | — | — | — | — | — | — | — | — | — | — | 是 | CLIP 77 token 截断后 Cityscapes 19 类只剩 4 个 |

## 2026-10-20 异常根因定位与代码修复记录

1. **为什么 SANA 共享 LoRA 的 Cityscapes mIoU 只有 0.009–0.017，而 Qwen-Image-2.1 在 5k 步就有 0.235？**
   - **根因 A（SANA LoRA Target Modules 漏掉全部注意力层）**：`model_registry.py` 中 `sana-1600m` 原默认 `lora_target_modules` 仅为 `["linear", "proj_out", "conv_inverted", "conv_point"]`（共 42 层，全部在 FFN 卷积、`proj_out` 和 `time_embed.linear` 上），**自注意力 `attn1` 和文本交叉注意力 `attn2`（`to_q, to_k, to_v, to_out.0`）100% 被冻结**。单训 CS-19 specialist 时色板固定，FFN 卷积可强行背下 19 种固定颜色；但多任务共享且 Cityscapes 仅占语义样本 6.97%（311/4460，占总步数约 1.7%）时，冻结的 `attn2` 无法根据 Prompt 中的 `{"class": <R, G, B>}` 图例动态路由颜色，FFN 卷积在多色板与黑白掩码间相互抵消。而 `qwen-image-2.1` 对 `to_q, to_k, to_v, to_out.0` 全部挂载了 LoRA，且源图进入 Qwen3-VL 视觉 token。
   - **根因 B（`sana-1600m` 漏设 `text_encoder_max_length=300`）**：`model_registry.py` 中仅 `sana-600m-512px` 设置了 `text_encoder_max_length=300`，而 `sana-1600m` 继承了 `ModelConfig` 默认值 `77`，导致训练与推理时 Gemma-2 tokenizer 在 77 token 处截断，Cityscapes 19 类 RGB 图例被切掉约 3/4。
   - **已落地修复**：已在 `src/open_vision_banana/model_registry.py`、`configs/experiments/sana_1600m_shared_baseline.yaml`、`scripts/train_sana_512.py`、`scripts/infer_sana_512.py` 中为 SANA 补齐 `text_encoder_max_length=300` 及 `to_q, to_k, to_v, to_out.0` 注意力 LoRA 目标模块。
2. **为什么所有模型（SANA、Qwen-Image-2.1）的 NYU / DIODE 度量深度闸门全部失败，而尺度对齐后 δ1 升至 0.37–0.46？**
   - **根因（训练 Target 编码与评测解码契约不一致 + Prompt 不一致）**：SNVC 训练 pack 中的 5630 条深度 target 是**逐图归一化的相对深度图（近处更亮灰度/伪彩）**，且训练 Prompt 为 `"Generate a relative depth map..."`；而 `ManifestVisionDataset` 在 `depth_curve: metric_power` 下直接原样读取 PNG，评测时 `evaluate_paper_benchmarks.py` 用论文 `metric_power`（$\lambda=-3, c=10/3$）+ `cube_edge` 逆变换解码为米数。在 `metric_power` 逆变换 $d = 10((1-u)^{-0.5}-1)$ 下，逐图归一化图像中 $u \in [0.5, 1.0]$ 的像素会被解码为 $4.14\text{m} \to \infty$，导致室内场景尺度严重膨胀（NYU AbsRel $\approx 2.0$，DIODE AbsRel $200\sim 5500$，NYU $\delta_1 \approx 0.066$，与单任务相对灰度 specialist `ovb-fix_ctrl_depth` 的 `0.066 / 1.990 / aligned 0.427` 完全一致）。
   - **已落地诊断工具**：新增 `validate_manifest_contract`（`src/open_vision_banana/data/validation.py`）与 `scripts/diagnose_shared_baseline.py`，在开训前自动拦截「无注意力 LoRA」「`max_length` 截断语义图例」「`metric_power` 混用相对深度 Prompt 或灰度/非 cube-edge 目标图」三类契约错误。

## 2026-10-22 第三轮论文对齐修复记录

对合并后的 `main` 重新逐项核对论文（arXiv:2604.20329v3）后落地了四项修复；前两项是方法层偏差，后两项是训练配方与零样本协议层面的补齐。

1. **指代分割 Prompt 训练 / 评测再次错配（回归）**。主线合入的 RES 契约在推理端输出 `Segment {x} in solid white. Render all other pixels black.`，而 `build_rea2seg_manifest.py` 训练用的是 Fig. 4 (c)/(d) 模板并被追加了一句 `Other objects are black.`。现在三处统一为论文 Fig. 4 (c)/(d) 逐字模板（白前景 / 黑背景）：`referring_contract.BINARY_REFERRING_TEMPLATE`，`PromptBuilder.REFERRING_TEMPLATES` 恢复为论文原句，表达式去掉前导冠词避免 "The the"。旧措辞的 RES manifest 仍可加载（`is_binary_referring_prompt`），但不再生成。
2. **Stage 5 Eq. (6) 面积基准**。论文用同一个符号 $A(\cdot)$ 写联合包围盒与两分量面积，`InstanceDecodingConfig.spatial_merge_area_basis` 默认取 `bbox`；原先的像素计数读法保留为 `pixel`（更严格、更少合并）供消融。字段已进入 `to_dict()` 契约。
3. **零样本审计补 image-id 级排除**。RefCOCOg UMD val 图像是 COCO train2014 文件，仅按路径关键词审计拦不住 COCO 来源的语义 / 实例数据。`build_qwen_baseline_manifest.py --exclude-image-ids refs(umd).p` 现在按 COCO 文件名解析 id 并剔除，只匹配 COCO 命名（`COCO_train2014_000000xxxxxx.jpg` / 12 位数字），Hypersim / Cityscapes 帧号不会误伤。
4. **`*_paper_aligned.yaml` 接入生成保留混训**。两份 Qwen paper_aligned 配置的 `tasks` 加入 `generation`，`vision_ratio: 0.5`（论文只说视觉数据占比 "very low"，未给数字，0.5 是 LoRA 预算下的折中，不是论文常数）。`scripts/build_generation_replay_manifest.py` 从带 caption 的图像集（`--kind t2i`）或编辑三元组（`--kind edit`）生成 `generation` 行；对 Qwen-Image-Edit / 2.1 这类以源图为条件的编辑底座，训练器会在加载模型前拒绝 `input == target` 的 T2I 式 replay 行（否则训练的是恒等重建而非生成），必须用 `--kind edit`。

5. **实例分割训练数据源（COCO / LVIS）**。SNVC 没有实例子集，新增 `scripts/build_instance_manifest.py`：对每个 (image, category) 对渲染黑底、每实例一个独立颜色的 target（调色板两两 RGB 距离 ≥ 60，远大于解码器 τ=14；大实例先画、小实例覆盖在上），Prompt 用论文 Fig. 3 模板 `PromptBuilder.build_instance_prompt(category)`。跳过有歧义的监督：COCO `iscrowd` 区域所在的类别对、LVIS `not_exhaustive_category_ids` 中的类别（该类未标全）；`--exclude-image-ids refs(umd).p` 复用第 3 条按 id 剔除 RefCOCOg 评测图（COCO 2017 与 2014 的 image id 一致）。单测验证渲染出的 target 经 `SegmentationDecoder.paper_reproduction()` 五阶段解码能精确恢复实例数（含遮挡）。训练器对 `instance` target 改用 NEAREST 重采样，避免双线性在实例边界混出伪颜色。两份 paper_aligned 配置 `tasks` 加入 `instance`，论文五任务至此在训练配方中齐全。
