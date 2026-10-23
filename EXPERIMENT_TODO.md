# Open-Vision-Banana 实验 TODO（协作者版）

> 日期：2026-10-23 ｜ 仓库：`isLinXu/open-vision-banana` `main`（#47、#48 已合入）
> GitHub 跟踪 issue：[#56](https://github.com/isLinXu/open-vision-banana/issues/56)；每个实验对应一个子 issue（见下表），**请在对应 issue 里勾选进度**。
> 完整设定、命令和闸门定义在仓库根目录 [`exp_setting.md`](exp_setting.md)；本文件是它的任务清单摘要。

## 一句话现状

代码层面已经完成：公开数据管线（下载 → 预处理 → 合并 → 严格零样本审计）、六任务统一训练器、三道机器闸门、pin 住的推理协议、论文逐项审计（58 条可核对陈述，49 完全一致、0 实现错误）。本地全量测试 **1611 passed, 0 failed**。
**剩下的全部是 GPU 上的实验**，按下面顺序做。

## 开跑前（任何人第一次上 GPU 机器都要做）

- [ ] `git pull`；`pip install -e ".[all]"`；再装 `peft accelerate diffusers wandb h5py pyarrow pycocotools ijson scipy modelscope`
- [ ] `export HF_ENDPOINT=https://hf-mirror.com`；`modelscope login`（OpenDataLab 的 Hypersim / ADE20K 需要）
- [ ] `PYTHONPATH=src:. pytest tests/ -q` —— 预期 **1611 passed**。如果看到约 35 个失败，是缺上面那几个包，不是基线
- [ ] 读一遍 `exp_setting.md` §0（硬件：1×80 GB GPU，≥300 GB 磁盘用于 E1）

## 适用于每一行结果的规则（违反任一条，数字不进结果表）

1. **推理协议固定**：`--num-inference-steps 50 --guidance-scale 4.0 --seed 42 --resolution 512 --native-resolution`。评测脚本默认的 1 步是 smoke 设置，**不能出现在结果表里**。每个数字旁边抄上评测 JSON 里的 `metadata.inference_protocol`；协议不同的数字不放同一张表。
2. **零样本 manifest 只能用** `build_qwen_baseline_manifest.py --protocol strict`（默认）。`in_domain` 的 run 必须标注，永远不报成零样本。
3. **没有 `GEMINI_API_KEY` 时**，T3（SA-Co cgF1 / IL_MCC）和 T5（ReasonSeg gIoU）只能报成"诊断值"；只有正样本 pmF1 是协议内的。
4. **生成保留**是对底座参照集的 FID / LPIPS / CLIP **代理**；论文指标是人工成对评测。报告时写"generation proxy"。
5. `*_aligned` 深度指标是诊断值，不是米制深度。
6. 状态写成 `训练 … / 结论 …` 两段；小于 E1 噪声下限的差异一律写"无显著差异"。
7. 论文 Table 1–7 的数字是参照不是目标：底座、内部数据、人工评测都是闭源的（见 `docs/PAPER_ALIGNMENT_AUDIT_2026-10-22.md`）。

## 实验清单（按顺序）

### E-1 旧 checkpoint 证据补齐 —— 仅评估，约 1 GPU 天 ｜ [#49](https://github.com/isLinXu/open-vision-banana/issues/49)
目的：旧结果表里有"训练完成但结论未完成"的行（外部评审第 3–5 项）。
- [ ] Qwen-Image-2.1 `uniform` **30k 全量评测**（`khazic/ovb-qwen_image_21_shared_uniform/checkpoint-30000_final`）：Cityscapes / NYU / DIODE / RefCOCOg / ReasonSeg
- [ ] T1 `uniform`、T2 `sqrt` 两个 SANA checkpoint 的 **DIODE 法向**（预设裁决规则的第三项指标）
- [ ] 按预设规则裁决：`sqrt` 要在 {Cityscapes mIoU, NYU δ1, DIODE mean°} 里 ≥2 项优于且无一退步才替换 `uniform`；否则保持"候选"
- [ ] **修复后 SANA 对照**：用 `--protocol in_domain` 重建 T1 manifest（含 Cityscapes train），重训 `sana_1600m_shared_baseline.yaml`（注意力 LoRA + 300 token），同协议评测
- [ ] 更新 `docs/ABLATION_BASELINE.md`
- **完成判据**：T1/T2/T7 行的结论状态不再因缺评测而 `pending`

### E-0 底座零训练对照 —— 仅评估，约半 GPU 天 ｜ [#50](https://github.com/isLinXu/open-vision-banana/issues/50)
目的：论文主张"微调只是解锁底座已有能力"；没有这一行，E1 的分数无法归因。
- [ ] 不加 `--lora-path`，用 E1 相同 prompt / 解码器 / 协议评测 Qwen-Image-2.1（每集 200 样本够用）
- [ ] `check_experiment_gates.py eval --require AbsRel,cIoU` 通过
- [ ] 记入 `ABLATION_BASELINE.md` 作为"零训练"行（预期解码大多失败——这正是要证明的）
- **完成判据**：之后每张表都有 E-0 行

### E0 真实数据管线 smoke —— 约半天，~30 GB ｜ [#51](https://github.com/isLinXu/open-vision-banana/issues/51)
目的：公开数据管线第一次在真实底座上跑通。**没有质量目标**，只看闸门。
```bash
bash scripts/run_e0_smoke.sh --dry-run          # 先审一遍命令
PYTHON=$(which python) bash scripts/run_e0_smoke.sh
```
- [ ] 下载 `0 failed`，Hypersim zip 过 `unzip -t`
- [ ] 六个 prepare 脚本跑完；`gates_data.json` 0 FAIL
- [ ] 500 步训练完成；`gates_run.json` 0 FAIL
- [ ] 推理抽检：深度图非常数，指代掩码至少一张非空
- [ ] 50 样本评测（50 步）过 `gates_eval.json`
- [ ] **步数敏感度**：`INFER_STEPS=20 bash scripts/run_e0_smoke.sh --from eval`，比较两份 JSON，决定 E1 用 20 步还是 50 步，**写回 `exp_setting.md` E1.3**
- [ ] 把 fixture 测不出的真实情况记下来：下载速度、显存、真实场景的 Hypersim 法向朝向、OmniEdit 实际 schema
- **完成判据**：三个闸门 JSON 全部 0 FAIL；步数决定已写入 E1.3

### E1 六任务主实验 —— Qwen-Image-2.1，30k 步，约 2–3 GPU 天，~300 GB ｜ [#52](https://github.com/isLinXu/open-vision-banana/issues/52)
**数据（E1.1）**
- [ ] 全量第一档：60 个 Hypersim 场景、6 个 TartanAir chunk、8 个 OmniEdit 分片、ADE20K、COCO-Stuff、COCO+LVIS、RefCOCO/+
- [ ] prepare（Hypersim `semantic_variants 2`，ADE20K `--variants 2`），`merge --cap 40000 --val-percent 2`，strict 审计
- [ ] `check_experiment_gates.py data` 通过；记录每任务行数

**训练（E1.2）**
- [ ] `--print-effective-config` 过目（六任务、`vision_ratio` 0.5、optimizer 块、preflight 干净）
- [ ] 30k 步，`save_every` / `eval_every` 2500；loss 始终有限；`check_experiment_gates.py run` 通过
- [ ] 两份审计 JSON + Hypersim `manifest.json` 拷进 run 目录

**评测（E1.3）**——协议：50 步（或 E0 决定的 20）、CFG 4.0、seed 42、512 画布、native 打分
- [ ] 5k / 15k / 30k 的 `_best` 各评一次：Cityscapes / NYU / DIODE / ETH3D / RefCOCOg / ReasonSeg
- [ ] 30k 再评一次 `--guidance-scale 0.0`（分数里有多少是 CFG 贡献）
- [ ] 30k 再评一次 `--seed 43` → 与三个 checkpoint 的离散度一起构成**噪声下限**
- [ ] 生成代理：先用底座在 GenAI-Bench prompt 上生成参照集，再评 checkpoint；ImgEdit-Bench 编辑侧同样两步
- [ ] 有 Gemini key：SA-Co Gold（HF gated，需申请）cgF1 / IL_MCC / pmF1，ReasonSeg 带改写；没有则标"诊断"

**内部成功标准**（不是论文数字；达不到是发现，不是失败）：Cityscapes mIoU ≥ 0.30 @30k · NYU δ1 > 0.45（常数深度基线 0.352）· RefCOCOg cIoU ≥ 0.35 · 生成 FID ≤ 1.3× 底座
- **完成判据**：`ABLATION_BASELINE.md` 结果行带 `inference_protocol`、噪声下限、E-0 并列，状态 `训练 done / 结论 done`

### E2 + E2b 消融 —— 5 × 10k 步 ｜ [#53](https://github.com/isLinXu/open-vision-banana/issues/53)
**E2 `vision_ratio`**（论文只说"very low"，没给数值）
- [ ] 0.2 × seed {42, 43}；0.8 × seed {42, 43}；E1 的 10k 作为 0.5 点
- [ ] 全部按 E1.3 协议评测，**含生成代理**（这个消融就是关于生成保留的）
- [ ] 裁决：某个比例要在 {Cityscapes mIoU, NYU δ1, DIODE mean°} 里 ≥2 项超过噪声下限且无退步才算胜出；否则"无显著差异"

**E2b 数据量**（论文："a small amount of vision task data"）
- [ ] `merge --cap 10000`（E1 的 1/4），同审计同配置，10k 步
- [ ] 与 E1 @10k 比较；写明视觉数据减到 1/4、replay 不变时各指标掉多少
- **完成判据**：一张 5+1 行的表，每行带协议和 seed，结论尊重噪声下限

### E3 附录 A 解码器 2×2 —— 仅评估 ｜ [#54](https://github.com/isLinXu/open-vision-banana/issues/54)
论文留下两处读法：Eq.3（像素只需匹配邻居，还是也要匹配 seed）× Eq.6（bbox 面积还是像素面积）。论文对齐默认 = `pixel_only` + `bbox`。
- [ ] 实例预测**只生成一次**并缓存；四种解码在缓存的 RGB 上跑
- [ ] 报四格 pmF1（有 Gemini 则加 cgF1）；默认格就是 E1 报的数
- [ ] 若非默认格明显更好，默认数字不改，另一格作为消融报告——不要悄悄切换
- **完成判据**：2×2 表入 `ABLATION_BASELINE.md`；`PAPER_INSTANCE_DECODING_CONFIG` 不变（除非论文原文支持）

### E4 新数据续训 —— 等第二档数据 ｜ [#55](https://github.com/isLinXu/open-vision-banana/issues/55)
- [ ] 新数据层走同一条管线到 strict 审计 + data 闸门（`data/qwen_baseline_v2`）
- [ ] `--resume <E1 30k_best> --resume-weights-only --steps 15000`（优化器 / warmup / step 归零；LoRA rank 与目标模块必须一致）
- [ ] E1.3 评测；与 E1 @30k 用噪声下限比较
- [ ] 一次只换一层数据，否则指标变化无法归因
- **完成判据**：结果行标注数据层（审计里的 `manifest_sha256`）与父 checkpoint

## 不在实验范围内

- GitHub Actions 因账号账单被阻塞，本地测试是权威信号
- 不复跑 SAM 3 / Depth Anything 3 / Lotus-2 等基线，直接引用论文表格

## 关键文件索引

| 要做什么 | 看哪里 |
|:--|:--|
| 完整实验设定与命令 | `exp_setting.md` |
| 数据来源、许可、每个源的 prepare 命令 | `docs/DATASETS.md`，`configs/datasets/sources.tsv` |
| 结果表与历史根因记录 | `docs/ABLATION_BASELINE.md` |
| 论文逐项对齐审计 | `docs/PAPER_ALIGNMENT_AUDIT_2026-10-22.md` |
| 一键 E0 | `scripts/run_e0_smoke.sh`（`--dry-run` / `--from <stage>`） |
| 机器闸门 | `scripts/check_experiment_gates.py data|run|eval` |
| 论文基准评测 | `scripts/evaluate_paper_benchmarks.py`（记得传协议参数） |
| 操作手册 | `AGENT.md`，`.claude/skills/vision-banana-train/SKILL.md` |
