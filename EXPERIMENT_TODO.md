# Open-Vision-Banana 实验计划与发布清单（协作者版）

> 日期：2026-10-23 ｜ 仓库 `isLinXu/open-vision-banana` `main`（#47、#48 已合入）
> GitHub 总览 issue：[#56](https://github.com/isLinXu/open-vision-banana/issues/56)；每个实验 / 发布项各有子 issue，**进度在 issue 里勾选**。
> 精确命令与闸门定义见 [`exp_setting.md`](exp_setting.md)；报告骨架见 [`docs/REPORT_OUTLINE.md`](docs/REPORT_OUTLINE.md)。本文件回答"做什么、为什么做、做到什么程度算完成、最后交付什么"。

---

## 0. 我们要回答的问题（报告的主线）

这个项目的目标**不是**复现 Nano Banana Pro 的数字（闭源，不可能），而是用开源底座检验论文的四个可检验主张，并把方法层的实现开源。报告的每一节都应能回到下面某一条：

| # | 论文主张（原文位置） | 我们的检验方式 | 对应实验 |
|:-:|:--|:--|:--|
| Q1 | 把视觉任务参数化为 RGB、用 prompt 选择任务，**一个权重**能做五个视觉任务（§2、§3） | 同一个 LoRA checkpoint 在五个任务基准上评测；与"每任务单独训"的 specialist 比 | E1、E-1 |
| Q2 | instruction-tuning 只是**解锁**底座已有的表征（§1、§4） | 零训练底座（E-0）vs 微调后（E1）；训练数据量减到 1/4（E2b）后掉多少 | E-0、E1、E2b |
| Q3 | 以**很低比例**混入生成数据即可**不损失生成能力**（§2） | `vision_ratio` 0.2/0.5/0.8 下视觉指标与生成代理的权衡曲线 | E2 |
| Q4 | 论文的编码 / 解码 / prompt 契约是**可复现、可解码**的（§3.1–3.2、附录 A） | 逐项对齐审计（已完成，58 条）+ 解码器 2×2 消融 + 真实输出上的解码失败率 | 审计、E3、E1 |
| Q5 | 结果在**严格零样本**协议下成立 | strict manifest 审计 + image-id 交集 = 0 的证据链 | 全部 |

**我们不能回答的**（报告里要写明）：论文的绝对数字是否可达（底座不同）、人工成对生成评测（没有人力）、in-house 标注数据的作用。

---

## 1. 开跑前（每台 GPU 机器第一次都要做）

- [ ] `git pull`；`pip install -e ".[all]"`；再装 `peft accelerate diffusers wandb h5py pyarrow pycocotools ijson scipy modelscope`
- [ ] `export HF_ENDPOINT=https://hf-mirror.com`；`modelscope login`（OpenDataLab 的 Hypersim / ADE20K）
- [ ] `PYTHONPATH=src:. pytest tests/ -q` → 预期 **1611 passed**（约 35 个失败 = 缺上面的包，不是基线）
- [ ] 记下 `git rev-parse HEAD`、`pip freeze > env.txt`、`nvidia-smi`，放进 `output/_environment/`——报告的"复现信息"节直接引用
- [ ] 申请 `GEMINI_API_KEY`（T3/T5 协议内评测需要）；SA-Co/Gold **标注**已可从 ModelScope 匿名下载（`bash scripts/download_datasets.sh --only saco-gold`，474 MB，PR #65），**图像**需注册免费 Roboflow 账号（MetaCLIP 6 子集）或填 Meta SA-1B 表单（`sa_co_gold.tar`）——Roboflow 当天可拿，Meta 表单可能等几天，**今天就提**

---

## 2. 适用于每一行结果的规则

1. **推理协议固定**：`--num-inference-steps 50 --guidance-scale 4.0 --seed 42 --resolution 512 --native-resolution`。脚本默认 1 步是 smoke 设置，**不进结果表**。每个数字旁抄 `metadata.inference_protocol`；协议不同的数字不进同一张表。
2. **零样本 manifest** 只用 `build_qwen_baseline_manifest.py --protocol strict`（默认）；`in_domain` 的 run 必须标注。
3. **无 `GEMINI_API_KEY`** 时 T3（SA-Co cgF1 / IL_MCC）、T5（ReasonSeg gIoU）只报"诊断值"；仅正样本 pmF1 协议内。
4. **生成保留**是对底座参照集的 FID / LPIPS / CLIP **代理**，报告写"generation proxy"，不写 App. D。
5. `*_aligned` 深度指标是诊断值。
6. 状态写 `训练 … / 结论 …`；差异小于 E1 噪声下限一律"无显著差异"。
7. 论文 Table 1–7 是参照不是目标。
8. **每个 run 目录必须保留**：`effective_config.json`、两份数据审计 JSON、`gates_*.json`、全部评测 JSON/CSV、`train.log`、`_best` 与 `_final` checkpoint。这是报告附录与开源发布的原材料，丢了等于没跑。
9. **每个数字只产生一次**：评测 JSON 是唯一真源，`ABLATION_BASELINE.md` 和报告里的表都从 JSON 生成（`scripts/check_experiment_gates.py eval` 已能读；表格生成脚本见发布清单 R1）。

---

## 3. 实验清单（按执行顺序）

每个实验：**目的 → 前置 → 任务 → 产出 → 完成判据 → 进报告哪一节**。

### E-1 旧 checkpoint 证据补齐 —— 仅评估，≈1 GPU 天 ｜ [#49](https://github.com/isLinXu/open-vision-banana/issues/49)
**目的**：外部评审指出旧结果表"训练完成 ≠ 结论完成"；补齐后旧实验才能作为报告里的 SANA 对照与"修复前后"对比。
**前置**：eval 集下载（`--group eval`）、Cityscapes 手动下载。
- [ ] Qwen-Image-2.1 `uniform` 30k 全量评测（`khazic/ovb-qwen_image_21_shared_uniform/checkpoint-30000_final`）：Cityscapes / NYU / DIODE / RefCOCOg / ReasonSeg
- [ ] T1 `uniform`、T2 `sqrt` SANA checkpoint 的 DIODE 法向
- [ ] 预设规则裁决 `sqrt` vs `uniform`
- [ ] 修复后 SANA 对照：`--protocol in_domain` 重建 T1 manifest，重训 `sana_1600m_shared_baseline.yaml`（注意力 LoRA + 300 token），同协议评测——这给出"注意力 LoRA + 文本长度"两项修复的效果
- [ ] 更新 `docs/ABLATION_BASELINE.md`
**产出**：旧 checkpoint 的完整评测 JSON；SANA 修复前/后对照行。
**完成判据**：T1/T2/T7 结论状态不再因缺评测 `pending`。
**进报告**：§5.4 底座对比（SANA vs Qwen）、§6 历史根因与修复。

### E-0 底座零训练对照 —— 仅评估，≈½ GPU 天 ｜ [#50](https://github.com/isLinXu/open-vision-banana/issues/50)
**目的**：Q2。没有这一行，E1 的提升无法归因于微调而非 prompt 工程。
- [ ] 不加 `--lora-path`，E1 相同 prompt / 解码器 / 协议评测 Qwen-Image-2.1（每集 200 样本）
- [ ] 同样评测 **Qwen-Image-Edit-2509** 底座（编辑底座先验不同，报告里要比较两者谁更适合做底座）
- [ ] `check_experiment_gates.py eval --require AbsRel,cIoU` 通过
- [ ] 保存 20 张底座输出图（每任务 4 张）——报告 Fig.「零训练 vs 微调」的左列
**产出**：两份 E-0 评测 JSON，底座输出图。
**完成判据**：后续每张表都有 E-0 行。
**进报告**：§5.1 主结果表的第一行；§5.5 "解锁"论证。

### E0 真实数据管线 smoke —— ≈½ 天，~30 GB ｜ [#51](https://github.com/isLinXu/open-vision-banana/issues/51)
**目的**：公开数据管线第一次在真实底座上跑通；决定 E1 的推理步数。**没有质量目标。**
```bash
bash scripts/run_e0_smoke.sh --dry-run
PYTHON=$(which python) bash scripts/run_e0_smoke.sh
INFER_STEPS=20 bash scripts/run_e0_smoke.sh --from eval      # 步数敏感度
```
- [ ] 下载 `0 failed`；Hypersim zip 过 `unzip -t`
- [ ] 六个 prepare 跑完；`gates_data.json` 0 FAIL
- [ ] 500 步训练；`gates_run.json` 0 FAIL；记录 **显存峰值、秒/步**（报告 §7 算力表）
- [ ] 推理抽检：深度非常数、指代掩码非空
- [ ] 50 样本评测（50 步）过 `gates_eval.json`；20 步再评一次，比较后**把步数决定写回 `exp_setting.md` E1.3**
- [ ] 记录 fixture 测不出的真实情况：下载速度、Hypersim 真实场景法向朝向比例（`manifest.json` 的 `normal_backfacing_frames`）、OmniEdit 实际 schema、每源准备耗时
**产出**：三个闸门 JSON、两份步数评测 JSON、算力记录。
**完成判据**：三闸门 0 FAIL；步数决定已写入 E1.3。
**进报告**：§4 数据管线（真实规模与耗时）、§7 算力与复现。

### E1 六任务主实验 —— Qwen-Image-2.1，30k 步，≈2–3 GPU 天，~300 GB ｜ [#52](https://github.com/isLinXu/open-vision-banana/issues/52)
**目的**：Q1、Q2、Q4、Q5 的主要证据；开源发布的主 checkpoint。
**数据（E1.1）**
- [ ] 全量第一档（60 Hypersim 场景、6 个 TartanAir v1 环境、8 OmniEdit 分片、ADE20K、COCO-Stuff、COCO+LVIS、RefCOCO/+）
- [ ] prepare → `merge --cap 40000 --val-percent 2` → strict 审计 → `gates_data` 通过
- [ ] 记录每任务 / 每来源的行数、图像数、许可（报告 §4 表；数据卡用）
**训练（E1.2）**——用 `scripts/run_e1_full.sh`：数据阶段在 CPU 节点 `--to merge`，GPU 节点 `RUN_DIR=... nohup bash scripts/run_e1_full.sh --from train`，第二块卡可 `--from eval --watch` 边训边评；脚本自动续训、无 key 时自动关 W&B
- [ ] `--print-effective-config` 过目并存档（runner 首次启动自动打印）
- [ ] 30k 步，`save_every` / `eval_every` 2500；loss 有限；`gates_run` 通过
- [ ] 保留 5k / 10k / 15k / 20k / 25k / 30k 的 `_best`（E2 需要 10k 点；报告需要学习曲线）
- [ ] W&B 或 `train.log` 导出：总 loss、每任务 loss、lr——报告 Fig.「训练曲线」
**评测（E1.3）**——协议：50 步（或 E0 决定）、CFG 4.0、seed 42、512 画布、native 打分
- [ ] 5k / 15k / 30k `_best`：Cityscapes / NYU / iBims-1（若拿到）/ ETH3D / DIODE-Indoor / RefCOCOg / ReasonSeg / DIODE 法向 / NYU 法向（若拿到 DSINE 包）/ VKITTI 法向（深度推导）
- [ ] 30k 再评 `--guidance-scale 0.0`（CFG 贡献）
- [ ] 30k 再评 `--seed 43`（噪声下限的一半来源）
- [ ] **解码失败率**：每个任务统计"解码器返回空 / 非法"的样本比例（评测 JSON 里有 `skipped_samples`）——Q4 的直接证据
- [ ] 生成代理：底座参照集（GenAI-Bench prompt）→ checkpoint；ImgEdit-Bench 编辑侧同样两步
- [ ] 有 Gemini：SA-Co Gold cgF1 / IL_MCC / pmF1、ReasonSeg 带改写；无则标"诊断"
- [ ] **每任务保存 8 张"输入 / 预测 RGB / 解码结果 / GT"四联图**（固定样本 id，E-0 与 E2 用同一批）——报告 Fig. 2–6 的素材
**内部成功标准**（不是论文数字；达不到是发现）：Cityscapes mIoU ≥ 0.30 @30k · NYU δ1 > 0.45 · RefCOCOg cIoU ≥ 0.35 · 生成 FID ≤ 1.3× 底座
**产出**：主 checkpoint（30k `_best`）、全部评测 JSON、噪声下限、四联图、训练曲线。
**完成判据**：结果行带协议、噪声下限、E-0 并列，状态 `训练 done / 结论 done`。
**进报告**：§5.1 主结果表、§5.2 学习曲线、§5.3 解码失败率、Fig. 2–6。

### E2 + E2b 消融 —— 5 × 10k 步 ｜ [#53](https://github.com/isLinXu/open-vision-banana/issues/53)
**目的**：Q3（混合比例）、Q2（数据量）。
**E2 `vision_ratio`**
- [ ] 0.2 × seed {42, 43}；0.8 × seed {42, 43}；E1 @10k 为 0.5 点
- [ ] E1.3 协议评测，**含生成代理**
- [ ] 画一张双轴图：横轴 `vision_ratio`，左轴视觉指标均值（归一化到 E1），右轴生成代理——报告 Fig.「权衡曲线」
- [ ] 预设规则裁决（≥2/3 指标超噪声下限且无退步）
**E2b 数据量**
- [ ] `merge --cap 10000`（1/4）；同配置 10k 步；与 E1 @10k 比
- [ ] 若时间允许加 `--cap 2500`（1/16）一个点，三点成曲线
**产出**：5+1（或 +2）行表、权衡曲线图。
**完成判据**：结论尊重噪声下限。
**进报告**：§5.6 消融。

### E3 附录 A 解码器 2×2 —— 仅评估 ｜ [#54](https://github.com/isLinXu/open-vision-banana/issues/54)
**目的**：Q4——论文留下的两处读法（Eq.3 seed 规则 × Eq.6 面积口径）对指标的影响。
- [ ] 实例预测只生成一次并缓存；四种解码跑缓存 RGB
- [ ] pmF1（有 Gemini 加 cgF1）四格；默认格 = E1 报的数
- [ ] 加一列"每图平均实例数 / 过分割率"，解释差异来源
**完成判据**：2×2 表入库；默认不改除非论文原文支持。
**进报告**：§5.7 解码器读法消融。

### E4 新数据续训 —— 等第二档数据 ｜ [#55](https://github.com/isLinXu/open-vision-banana/issues/55)
**目的**：数据质量 / 多样性对结果的边际贡献；为发布 v2 checkpoint。
- [ ] 新层走同一管线到 strict 审计
- [ ] `--resume <E1 30k_best> --resume-weights-only --steps 15000`
- [ ] 一次只换一层数据；E1.3 评测；噪声下限比较
**完成判据**：行标注 `manifest_sha256` 与父 checkpoint。
**进报告**：§5.8（若完成）。

### E5 定性与"野外"测试 —— 仅推理，≈½ GPU 天 ｜ [#58](https://github.com/isLinXu/open-vision-banana/issues/58)
**目的**：报告的图像证据；论文 Fig. 2–4、6–8 的开源对应物；Fig. 7 的"vibe test"。
- [ ] **开放词表语义**：10 张非 Cityscapes 图（COCO val / 自拍），prompt 用四种风格各一次，展示同一图不同风格输出一致——Fig. 2 对应
- [ ] **实例**：10 张多实例图，三种 Fig. 3 句式 + 非黑背景——Fig. 3 对应
- [ ] **指代**：10 条含外观 / 动作 / 非常规用途 / 多语言文字的表达——Fig. 4 对应
- [ ] **深度野外测试**：3–5 张手机照片，解码后的米数与可查证距离（卷尺 / 地图）对比，报 AbsRel——Fig. 7 对应
- [ ] **深度点云**：NYU / ETH3D 各 2 张，用 GT 内参反投影渲染两个视角——Fig. 6 对应
- [ ] **法向对比**：同一图上我们 vs Lotus-2 HF demo 输出（引用其 demo，注明非本仓库运行）——Fig. 8 对应
- [ ] **失败案例**：每任务 3 张典型失败 + 一句归因（实例合并 / 群体拆分 / 漏检，呼应论文附录 C）
**产出**：`output/figures/` 下按 fig 编号的 PNG 与生成脚本。
**完成判据**：每张图有固定 seed、prompt、checkpoint 记录，可一键重生成。
**进报告**：§5.9 定性结果、§6 失败分析、全部 Figure。

### E6 鲁棒性与跨任务迁移 —— 仅评估，≈1 GPU 天 ｜ [#59](https://github.com/isLinXu/open-vision-banana/issues/59)
**目的**：论文 §3.1 末段"cross-task transfer"与 Fig. 2 "various prompting styles" 的量化版本；检验训练期增强真的起作用。
- [ ] **prompt 风格鲁棒性**：Cityscapes val 用 `json_tuple` / `json_named` / `natural` / `hex` 四种 prompt 各评一次（同一图例）；四个 mIoU 的离散度 = 风格鲁棒性
- [ ] **颜色重映射鲁棒性**：Cityscapes 换三套随机图例颜色各评一次；mIoU 不随颜色变 = 模型读 prompt 而非背调色板（这是 SANA 时代失败的核心症状）
- [ ] **深度色图鲁棒性**：NYU 用 cube_edge / viridis / grayscale prompt 各评一次（对应解码器切换）
- [ ] **跨任务迁移**：在**语义** prompt 里放入指代式短语（"the man in pink shirt"），看能否分割——论文 Fig. 2b 右的现象
- [ ] **负样本行为**：SA-Co 负样本不经 Gemini 直接送模型，统计输出"接近全背景"的比例（论文说未训练此能力，我们量化一下）
**产出**：鲁棒性表（4 风格 × 1、3 颜色 × 1、3 色图 × 1）、迁移样例图、负样本统计。
**完成判据**：每项有数字和 ≥4 张样例。
**进报告**：§5.10 鲁棒性与迁移。

---

## 4. 发布清单（开源交付物）

### R1 代码与数据工件 ｜ [#60](https://github.com/isLinXu/open-vision-banana/issues/60)
- [ ] **checkpoint 上传 HF**：E1 30k `_best`（LoRA adapter + `metadata.json` + `effective_config.json`），命名 `ovb-qwen-image-2.1-six-task-v1`；E2 的 0.2/0.8 和 E2b 作为 ablation 子目录
- [ ] **模型卡**：底座、训练数据（来源 + 许可 + 行数 + `manifest_sha256`）、推理协议、每基准数字（含 E-0 对照和噪声下限）、已知失败模式、"非 Nano Banana Pro 复现"声明、许可（底座 Qwen 的许可 + 数据许可的交集——Hypersim 研究用途限制会传导到 checkpoint）
- [ ] **数据卡**：`configs/datasets/sources.tsv` 的表格化版本 + 每源的 prepare 命令 + 零样本审计产物（审计 JSON 直接附上）
- [ ] **评测结果包**：`results/` 目录，每个 run 一个子目录，内含评测 JSON/CSV、`inference_protocol`、闸门 JSON；加 `scripts/render_results_tables.py` 从 JSON 生成 Markdown 表（报告和 README 都从它生成，避免手抄）
- [ ] **一键复现脚本**：`scripts/run_e1_full.sh`（仿 `run_e0_smoke.sh`，E1 全流程 + 闸门）；README "Reproduce E1" 一节
- [ ] **环境锁定**：`requirements-lock.txt`（`pip freeze`）、CUDA / 驱动版本、`git` commit、HF 模型 revision
- [ ] **Demo**：`scripts/demo.py` 能加载发布的 adapter，六任务各出一张图；Gradio 可选
- [ ] 许可审查：代码 License、各数据集许可对 checkpoint 再分发的约束、Qwen-Image 许可——写进 README 与模型卡

### R2 技术报告 ｜ [#64](https://github.com/isLinXu/open-vision-banana/issues/64)（骨架见 `docs/REPORT_OUTLINE.md`）
- [ ] §1 摘要与主张（Q1–Q5 各一句结论）
- [ ] §2 与论文的关系：对齐审计摘要（58 条 / 49 一致 / 0 错误）+ 不可复现项清单
- [ ] §3 方法：编码契约、prompt、解码器（引用审计表，不重写公式）
- [ ] §4 数据：来源表、许可、零样本审计证据、与论文 in-house 数据的差异
- [ ] §5 实验：5.1 主结果（E1 + E-0 + specialist 对照）· 5.2 学习曲线 · 5.3 解码失败率 · 5.4 底座对比（SANA vs Qwen，含修复前后）· 5.5 "解锁"论证 · 5.6 消融（E2/E2b）· 5.7 解码器 2×2 · 5.8 E4 · 5.9 定性 · 5.10 鲁棒性与迁移
- [ ] §6 失败分析与历史根因（相对深度契约、注意力 LoRA、77 token、混合 batch flow、背景色——评审六项的来龙去脉）
- [ ] §7 算力、复现信息、噪声下限估计方法
- [ ] §8 局限与诚实边界（底座差距、无人工评测、Gemini 依赖、单数据档）
- [ ] 附录：完整配置、每基准样本数与协议、全部评测 JSON 索引
- [ ] 内部评审：至少一位未参与实验的人按"每个数字能否回溯到 JSON"过一遍

---

## 5. 算力与时间预算（估算，E0 后更新）

| 实验 | GPU 时 | 磁盘 | 可并行 |
|:--|--:|--:|:--|
| E-1 | ~24 h（含 SANA 重训 ~12 h） | 已有 | 与 E0 并行 |
| E-0 | ~8 h | eval 集 | 与 E0 并行 |
| E0 | ~4 h + 下载 | 30 GB | — |
| E1 | ~30 h 训练 + ~12 h 评测 | 300 GB | 评测可在第二块卡 |
| E2 + E2b | 5 × ~10 h + 评测 | 同 E1 | 两卡并行 |
| E3 | ~4 h | — | 与 E2 并行 |
| E5 + E6 | ~12 h | — | E1 后任意时间 |
| **合计** | **≈150–180 GPU 时** | | 单卡约 7–8 天，双卡约 4–5 天 |

---

## 6. 风险与预案

| 风险 | 影响 | 预案 |
|:--|:--|:--|
| Gemini API 不可用 / 版本漂移 | T3、T5 只能报诊断值 | 规则 3；报告明示；pmF1 仍可报 |
| SA-Co Gold 图像拿不到（Roboflow / Meta 表单） | E3、T3 用 COCO val 实例代替 | 标注已有（ModelScope）；E3 已写 fallback；模型卡注明 |
| SA-Co 三人标注 oracle 协议未实现 | cgF1 偏悲观、与论文不可比 | 新 issue（三 pass oracle）在 E1 评测前完成 |
| Qwen-Image-2.1 显存 > 80 GB | E1 跑不起 | `gradient_checkpointing`、分辨率 448（32 倍数）、或切 Qwen-Image-Edit-2509 |
| E1 内部标准未达 | 不是失败 | 写进 §6；检查 E-0 → E1 的提升是否仍显著 |
| 下载源失效 | 管线中断 | 注册表换镜像即可（已有先例：UNC、ADE20K、COCO-Stuff） |
| 数据许可限制再分发 | checkpoint 发布受限 | R1 许可审查前置；必要时只发 adapter + 训练配方 |

---

## 7. 不在范围内

- GitHub Actions（账号账单阻塞，本地测试为权威信号）
- 复跑 SAM 3 / Depth Anything 3 / Lotus-2 基线（引用论文表格与各自论文）
- 人工成对生成评测
- 复现 Nano Banana Pro 的绝对数字

## 8. 关键文件索引

| 要做什么 | 看哪里 |
|:--|:--|
| 精确命令、闸门、协议 | `exp_setting.md` |
| 报告骨架与每节要填的表 | `docs/REPORT_OUTLINE.md` |
| 数据来源、许可、prepare 命令 | `docs/DATASETS.md`，`configs/datasets/sources.tsv` |
| 结果表与历史根因 | `docs/ABLATION_BASELINE.md` |
| 论文逐项审计 | `docs/PAPER_ALIGNMENT_AUDIT_2026-10-22.md` |
| 一键 E0 / 机器闸门 | `scripts/run_e0_smoke.sh`，`scripts/check_experiment_gates.py` |
| 论文基准评测 | `scripts/evaluate_paper_benchmarks.py`（必须传协议参数） |
| 操作手册 | `AGENT.md`，`.claude/skills/vision-banana-train/SKILL.md` |
