# Checkpoint 选择依据

调查日期：2026-10-03，Asia/Shanghai；main 基准 [`b490e2d3c91d5909de1ef9c9416768ec364ab884`](https://github.com/isLinXu/open-vision-banana/tree/b490e2d3c91d5909de1ef9c9416768ec364ab884)。**优先测试 fix_best3_s1／checkpoint-30000_final EMA，rank 32，SANA-1600M**；置信度中等，质量与本机成本尚未复现。

覆盖当日快照的 115 个公开仓库、93 个有结果 run、209 个 metrics；同日再核验 7 个候选的 revision、指标和配置，均与快照一致。[候选审计](evidence/checkpoint-candidate-audit-20261003.json)与[候选元数据](../../scripts/benchmark/configs/checkpoint-candidates.json)随仓库保存。分组／512px 来自此前读取的 [W&B 阶段报告](https://forge.coreweave.com/wandb/khazzz1c/open-vision-banana/reports/--VmlldzoxODA0Mjc1Ng==)；本轮未导出交互曲线，不保证之后新增结果已覆盖。

以下是历史上传值，非 Flickr30k 成绩。SANA 三任务条目均为 EMA、20 步，depth／normal／semantic 样本数 319／276／241；相同数量不证明相同协议。深度表示域、mIoU 聚合、manifest 与服务器代码尚未核验。

| 候选／checkpoint | 报告分组 | rank | mIoU ↑ | AbsRel ↓ | 法线角度 ↓ |
|---|---|---:|---:|---:|---:|
| fix_best3_s1／30000_final（主） | native-SANA 待核验 | 32 | 0.255735 | 0.398804 | 18.9633° |
| fix_best3_s42／30000_final（种子） | native-SANA 待核验 | 32 | 0.251056 | 0.395371 | 19.0103° |
| fix_no_minsnr／30000_final（单变量） | native-SANA 待核验 | 32 | 0.216162 | 0.486608 | 20.6823° |
| best3_rank64／27000_best（历史最高） | legacy-SANA 待核验 | 64 | 0.263032 | 0.389753 | 18.5369° |
| no_minsnr／30000_final（旧对照） | legacy-SANA 待核验 | 32 | 0.208650 | 0.519595 | 21.0138° |
| flux3_ctrl_seg_nominsnr_15k／15000_final | FLUX 分割专用 | 32 | 0.187148 | — | — |
| flux3_nominsnr_15k／15000_final | FLUX 三任务 | 32 | 0.165378 | 1.195346 | 22.5068° |

**事实：** [主候选指标](https://huggingface.co/khazic/ovb-fix_best3_s1/blob/54d4d2086153e85e7c32ce61ed82a1df6cdc7657/eval/checkpoint-30000_final_ema/metrics.json)在报告 native 组内分割最高，但深度略差于种子对照，不能称三任务全面最优。[legacy 高分](https://huggingface.co/khazic/ovb-best3_rank64/blob/4ac1c28bd54b090ce16ca113233c11f5db059ab5/eval/checkpoint-27000_best_ema/metrics.json)保留独立协议对照；不能直接交给当前默认流程。

主候选 final 相比其 17000 best，三个主指标更好，但 pixel accuracy 从 0.587101 降至 0.577727；best 后缀只代表验证 loss。[17000 指标](https://huggingface.co/khazic/ovb-fix_best3_s1/blob/54d4d2086153e85e7c32ce61ed82a1df6cdc7657/eval/checkpoint-17000_best_ema/metrics.json) **推断：** 主候选与种子对照值得先核验；不能由这些分数断言独立测试集冠军。

## 成本与实物核验

[SANA 官方卡](https://huggingface.co/Efficient-Large-Model/Sana_1600M_512px_diffusers)描述约 1.6B 骨干及 Gemma2 文本编码器／DC-AE；[FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)骨干为 12B。这是优先探索 SANA 的工程理由，不是实测速度结论。rank 32 在线 adapter 56,786,576B、rank 64 为 113,561,272B，骨干形状不变；rank 减半不代表推理快一倍。

上传 seconds 缺硬件、并发及计时边界，不用于成本排序；分割专用 FLUX 的 241 样本也不能与 SANA 的 836 个任务样本总耗时比较。缩图减少激活、不减少权重；6GB GPU／约16GB RAM 的完整 SANA 部署与卸载仍待测。[官方卸载说明](https://huggingface.co/docs/diffusers/main/en/optimization/memory)

同日已安全准备主候选 final EMA：84 张量／42 对、14,193,664 个 LoRA 参数，rank32／alpha64／rsLoRA；导出逐项等于 checkpoint EMA，全部与 online 不同，最大差 0.00265729。EMA 为 56,786,552B，SHA256 `6317a9a78c4df187677e3e6ffdce356adfb94a83998798c59610deb7695f12e4`；禁止下载后的缓存复跑 hash 不变。[实物证据](evidence/selected-checkpoint-preflight-20261003.json)包含固定远程来源及 hash，不依赖原机器路径。未下载基座或执行训练模型推理。

## 下一轮验收

- 固定 repo／revision／checkpoint／EMA 与基座 revision；按 provenance 校验来源，缺 EMA 必须失败，不回退 online。
- 取得服务器完整源码／patch 和实际导入路径；日志 f8aac5f 与 fix 前缀不足以证明 native 兼容。核对 [时间步／flow](https://github.com/isLinXu/open-vision-banana/blob/b490e2d3c91d5909de1ef9c9416768ec364ab884/src/open_vision_banana/training/model_backend.py#L449)、VAE 缩放、prompts、palette／深度／法线编码。
- 当前 main [默认深度 decoder](https://github.com/isLinXu/open-vision-banana/blob/b490e2d3c91d5909de1ef9c9416768ec364ab884/src/open_vision_banana/evaluation/evaluator.py#L217)与[历史 normalized 指标](https://github.com/isLinXu/open-vision-banana/blob/b490e2d3c91d5909de1ef9c9416768ec364ab884/scripts/eval_checkpoints.py#L62)不能盲用；同协议重测并保留原始输出。
- 先现代 CUDA、单图 batch1／256，再512／20步与带 GT manifest；记录加载、预测、解码、显存和 host RAM，失败不静默改配置。

当前 candidates 的 inference_contract_verified=false 仍有效，训练模型模板默认拒绝执行；EMA 准备成功不是推理验收。无像素 GT 的 Flickr30k 只做定性与成本，按 [README](README.md) 使用相对数据／权重／缓存路径；组件实测见 [testing.md](testing.md)。
