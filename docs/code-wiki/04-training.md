# 训练系统

[返回首页](README.md)

> 盘点基准：`main` 在 PR #37–#40 合入之后。本章只讲统一入口 `scripts/train.py` → `training.trainer.Trainer`；旧版 `train_sana_512.py` / `train_flux.py` / `InstructionTuner` 只在最后一节提及边界。

## 1. 调用链

```text
scripts/train.py
  └─ training.config.load_config(yaml) + apply_cli_overrides(argv)  → TrainingConfig（dataclass）
       └─ Trainer(config)
            ├─ ModelBackend(model_name, dtype, quantize, lora_*)      model_registry 加载 + PEFT LoRA
            ├─ _run_preflight()                                       validate_manifest_contract；error 则拒绝开训
            ├─ _build_datasets()                                      ManifestVisionDataset × 任务 → ConcatDataset
            │     └─ TaskWeightedSampler(strategy, vision_ratio)      一个全局抽样流，按 rank 分片
            ├─ AdamW(trainable_params, lr, weight_decay, betas, eps)  + warmup/cosine scheduler
            ├─ EMAModel（可选）  TrainingLosses  TrainingCallbacks
            ├─ _resume(path) / _resume_weights_only
            └─ train()
                 └─ 每步 _train_step(batch) → 梯度累积 → clip → step → EMA.update
                      ├─ backend.encode_image(input / target)            VAE → latents
                      ├─ backend.prepare_flow_batch(...)                 t、noisy、velocity target
                      ├─ backend.encode_text / encode_edit_text(prompt, image)
                      ├─ backend.forward(noisy, t, text, context_latents) 预测速度
                      └─ losses.compute_flow_loss + compute_aux_losses
                 ├─ 每 eval_every 步 _validate()  →  callbacks 记录 best
                 └─ 每 save_every 步 callbacks.on_checkpoint(...)
```

每次运行开头写 `output/<run>/effective_config.json`：合并后的全部字段、每任务样本数、采样概率、optimizer 块。它是这轮实验的可复现凭证，评测前先看它。

## 2. `TrainingConfig` 字段速查

[`training/config.py`](../../src/open_vision_banana/training/config.py)。YAML 分组 → dataclass 字段映射见文件内 `_YAML_SECTIONS`；每个字段都可用 `--kebab-case` CLI 覆盖，`--print-effective-config` 只打印不训练。

| 组 | 字段 | 含义 / 约束 |
|:--|:--|:--|
| model | `model_name`, `dtype`, `resolution`, `quantize` | Qwen-Image-2.1：分辨率必须是 32 的倍数且 ≥256，`batch_size` 必须为 1，量化加载被拒绝 |
| lora | `lora_rank`, `lora_alpha`, `rslora`, `lora_targets` | resume 时三者必须与 checkpoint 一致，否则 `ValueError` |
| data | `pair_manifest`, `data_dir`, `tasks`, `vision_ratio`, `task_sampling`, `task_sampling_weights`, `max_samples_per_task`, `val_max_samples_per_task`, `num_workers` | `tar_index` 字段仍被拒绝（自动发现 manifest 旁的 `tar_index.json`）；`vision_ratio < 1` 要求 `tasks` 含 `generation` |
| data（增强） | `depth_curve`, `depth_log_alpha`, `depth_colormaps`, `semantic_prompt_styles`, `semantic_color_remap_prob`, `aspect_preserving_crop` | 见[数据章 §5](03-data.md#5-训练期数据加载manifestvisiondataset) |
| training | `steps`, `batch_size`, `grad_accum`, `lr`, `weight_decay`, `adam_beta1`, `adam_beta2`, `adam_eps`（YAML `optimizer:` 块）, `max_grad_norm`, `drop_grad_norm`, `scheduler`, `warmup_steps` / `warmup_ratio`, `seed`, `gradient_checkpointing` | `steps` 是**绝对步数**（完整 resume 后只跑剩余） |
| flow | `img2img_flow`, `flow_sigma`, `logit_normal`, `flow_shift`, `min_snr`, `min_snr_gamma`, `cfg_dropout_rate` | SANA 用 linear_flow + shift；FLUX/SD3/Qwen 用 rectified flow |
| loss | `task_loss_weights`, `depth_edge_aux_*`, `normal_angular_loss_weight` | 辅助损失需要 `backend.decode_latents` |
| checkpoint | `output_dir`, `save_every`, `eval_every`, `log_every`, `ema_enabled`, `ema_decay`, `ema_warmup`, `resume`, `resume_weights_only` | |
| training | `skip_preflight` | 绕过 manifest 契约审计（error 仍记录到 `effective_config.json`） |
| logging | `wandb`, `wandb_project`, `wandb_run_name`, `device` | |

论文**写明**的只有数据侧契约（深度公式、prompt 字符串、Appendix A）；optimizer、lr、schedule、ratio、rank、分辨率全是仓库选择，`configs/experiments/*_paper_aligned.yaml` 的注释逐项标注了这一点。

## 3. `ModelBackend` 契约

[`training/model_backend.py`](../../src/open_vision_banana/training/model_backend.py) 是训练器与 `model_registry` 之间的唯一接口。任何新底座只需满足以下方法（`tests/test_trainer_system_all_tasks.py` 里的 `FakeBackend` 就是这个接口的最小实现）：

| 方法 | 作用 |
|:--|:--|
| `encode_image(images)` | `[-1,1]` 图像 → VAE latents（Qwen 2.1：64 通道、下采样 16，并应用 `latents_mean/std`） |
| `decode_latents(latents)` | 辅助损失 / 验证可视化 |
| `prepare_flow_batch(target, input, B, img2img_flow, flow_sigma, ...)` | 采样 `t`、构造 noisy latents 与速度目标；返回 `timesteps / noisy_latents / velocity_target / noise` |
| `encode_text(prompts, device, dtype)` | T2I 底座的文本条件 |
| `encode_edit_text(prompts, images, cfg_dropout_rate)` | 编辑底座：Qwen 2.1 把源图作为 **Qwen3-VL 视觉 token** 送入文本编码器（论文"natively image-conditioned"的对应物） |
| `apply_cfg_dropout(cond, rate, B)` | 以 `cfg_dropout_rate` 置空条件，保留推理期 CFG 能力 |
| `forward(latents, t, text_cond, B, guidance, context_latents)` | 预测速度；编辑底座按 [`edit_forward.edit_latent_layout`](../../src/open_vision_banana/training/edit_forward.py) 决定源图 latents 是 **序列拼接**（Qwen、Kontext）还是 **通道拼接**（InstructPix2Pix） |
| `peft_model` / `family` / `model_config` | 参数、flow 家族、注册表配置 |

`family` 决定 `FlowScheduler` 的时间步分布与 shift；`model_config.instruction_editing` 决定 `_train_step` 是否把源图 latents 作为上下文传入 `forward`。

## 4. 一步训练的张量流

以 Qwen-Image-2.1、分辨率 512、batch 1 为例：

```text
batch["image"]       [1,3,512,512] ∈ [-1,1] ──encode_image──► input_latents  [1,64,32,32]
batch["target_rgb"]  [1,3,512,512] ∈ [0,1]  ──×2−1──encode──► target_latents [1,64,32,32]
prepare_flow_batch: t ~ logit-normal(shift)；noisy = (1−t)·start + t·target，start = noise（编辑布局 / 纯生成行强制如此）或 input_latents（img2img_flow=True 的 T2I 底座）
encode_edit_text(prompt, image) ─► {encoder_hidden_states, mask, …}（源图视觉 token 已在其中）
forward(noisy, t, text, context_latents=input_latents) ─► v̂  [1,64,32,32]
loss = MSE(v̂, velocity_target) × task_loss_weights[task] × (min_snr 权重) + aux
```

`generation` 任务的 T2I 行 `input == target`；编辑行 `input ≠ target`，`_reject_identity_generation_replay` 在编辑底座上拒绝恒等行，防止 replay 教模型"什么都不改"。

## 5. 采样与混合

[`training/sampling.py`](../../src/open_vision_banana/training/sampling.py) `TaskWeightedSampler`：先按任务概率抽任务，再在任务内有放回抽样；一个全局随机流按 `rank / world_size` 分片，DDP 下各 rank 看到不同样本但同一任务序列。

- `task_sampling = proportional | uniform | sqrt`（按行数 / 均匀 / 行数开方）作用于**视觉任务之间**；
- `vision_ratio` 把总概率切成 `vision_ratio` 给视觉任务、`1 − vision_ratio` 给 `generation`。`effective_config.json` 的 `task_sampling_probabilities` 记录最终值；`tests/test_trainer_system_all_tasks.py` 断言 0.5 → generation 0.5 / 其余各 0.1。
- 每个 data pass 重新 `set_epoch`；resume 回到所在 pass 的开头而非精确 batch（对 LoRA 微调无实质影响）。

## 6. 验证、checkpoint 与恢复

- `_validate()` 在每个任务的 val 行上算 flow loss 均值；要求"有 val 的话每个任务都有"，否则构建期报错。
- `TrainingCallbacks.on_checkpoint` 写：

```text
output/<run>/
├── effective_config.json
├── checkpoint-{N}/          lora.pt（LoRA + optimizer + scheduler + EMA shadow + step + best_val_loss + selection_protocol）
│                            lora_adapter/（PEFT save_pretrained）  metadata.json
├── checkpoint-{N}_best/     验证最优（EMA 开启时保存 EMA 权重）
└── checkpoint-{N}_final/
```

- `_resume(path)`：完整恢复上述全部状态；`_selection_protocol()` 把 manifest sha256 等算进指纹，manifest 变了 `best_val_loss` 自动作废并告警。
- `resume_weights_only=True`：只加载 LoRA 权重，optimizer / scheduler / step 归零，EMA 从加载权重重新开始——**换数据、换 lr、换 schedule 时用这个**。
- RNG 不保存：恢复后用 `seed + rank + step` 重新播种，避免重放噪声。
- LoRA 结构不一致直接 `ValueError`，不做部分加载。

## 7. 分布式与硬件

- 统一训练器支持单进程或 `torchrun` DDP：每个 rank 加载同一 checkpoint，LoRA 梯度 all-reduce。**不接受** `--fsdp` / `--deepspeed`；旧版 `trainer_interface.py` 里的分布式类不能证明该入口支持它们。
- Qwen-Image-2.1 在 bf16、512 px、batch 1 下约需 60–70 GB；`gradient_checkpointing` 可降显存；`gradient_accumulation` 补有效 batch。
- SANA-1600M 可在 24 GB 上以 batch 4 训练；FLUX.1-dev 需 `quantize: 4bit`。

## 8. 旧版设施与边界

- `scripts/train_sana_512.py`、`scripts/train_flux.py`、`training/tuner.py`（`InstructionTuner`）、`config.py` 的 `VisionBananaConfig` 仍保留，用于复现早期 SNVC 实验；新实验一律用统一入口。
- `trainer_interface.py` 的 Native / DDP / FSDP / DeepSpeed / Accelerate 五后端属于旧版抽象，与 `Trainer` 无关。
- 已跑过的真实训练：SANA 与 FLUX 三任务（SNVC 数据）、Qwen-Image-2.1 共享 LoRA 30k 步（`docs/ABLATION_BASELINE.md` T7）。公开数据六任务配方的第一次真实运行按 [`exp_setting.md`](../../exp_setting.md) 的 E0 执行。
