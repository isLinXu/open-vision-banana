# 训练实验契约修复实施计划

> **For agentic workers:** 按任务逐项实施，使用 test-driven-development；保留已有改动，不自动提交。

**Goal:** 保证统一入口配置、任务采样和最佳权重导出与实验声明一致。

**Architecture:** 保留 dataclass、Trainer 和 callbacks；严格归一化配置，独立采样器，分离 online 续训权重与 EMA 推理导出。生成混训尚未实现，因此拒绝非 1 的 vision_ratio。

**Tech Stack:** Python 3.11、PyTorch、pytest、YAML、PEFT 接口的 CPU 测试替身。

## 已确认的设计

- 默认 proportional 保持遍历行为；uniform/sqrt 使用有放回全局抽样并按 rank 分片。
- 配置优先级为默认值、YAML、显式 CLI；未知键及未实现能力报错，不自动转换 epochs 或猜测 manifest。
- 验证按任务独立截取固定子集，采用按样本累计、按任务宏平均的 flow loss，不冒充论文指标。
- best adapter 使用验证时权重；lora.pt 始终保存 online 参数与 optimizer，元数据标明两个权重来源。
- 当前阶段不下载模型、不做真实训练、不启用生成混训、不自动提交。

## Task 1：有效配置

Files: `training/config.py`、`scripts/train.py`、`training/model_backend.py`、`tests/test_unified_training_config.py`。

- [ ] 先写字符串权重、真实 YAML、CLI 双向布尔覆盖、未知键、未实现混训回归。
- [ ] 执行 `PYTHONPATH=src .venv/bin/python -m pytest tests/test_unified_training_config.py -q`，确认功能性失败。
- [ ] 修复解析与 targets 传递；增加 `--print-effective-config`，运行前校验并输出 JSON。
- [ ] 同一命令全部通过，运行关联入口/模型测试。

```python
assert load_config(config_path=path).task_loss_weights == {
    "depth": 1.0, "normal": 2.0, "semantic": 1.5
}
```

## Task 2：任务采样与严格数据

Files: 新增 `training/sampling.py`、`tests/test_unified_task_sampling.py`；集成 `training/trainer.py`。

- [ ] 写 100:4:1 数据的概率、种子、epoch、DDP 序列分片测试。
- [ ] 执行 `PYTHONPATH=src .venv/bin/python -m pytest tests/test_unified_task_sampling.py -q`，确认失败。
- [ ] 实现策略概率和确定性全局抽样；默认比例遍历继续使用原 sampler。
- [ ] 数据缺失、任务为空、drop_last 导致空 loader 明确失败；记录期望与实际采样计数。
- [ ] 同一命令及 `tests/test_unified_trainer_ddp.py` 通过。

```python
# Task probabilities, not per-sample probabilities.
expected = {"depth": 1 / 3, "normal": 1 / 3, "semantic": 1 / 3}
```

## Task 3：EMA 与验证

Files: `training/callbacks.py`、`training/trainer.py`、`tests/test_unified_trainer_ema_checkpoint.py`、`tests/test_unified_trainer_validation.py`。

- [ ] 写 online=1、EMA=3 的 checkpoint 测试及任务覆盖、宏平均、异常恢复测试。
- [ ] 执行 `PYTHONPATH=src .venv/bin/python -m pytest tests/test_unified_trainer_ema_checkpoint.py tests/test_unified_trainer_validation.py -q`，确认失败。
- [ ] 保存 online 续训状态后临时应用 EMA 导出 adapter，finally 恢复；导出失败不提交最佳分数。
- [ ] 验证分任务固定预算、隔离 RNG、恢复模式；新协议下旧 best 分数不得沿用比较。
- [ ] 验证导出与完整 resume 测试通过。

```python
assert checkpoint["lora_state_dict"]["weight"].item() == 1
assert exported_adapter["weight"].item() == 3
assert model.weight.item() == 1
```

## Task 4：验收与后续资产

- [ ] 执行 unified trainer、prompt roundtrip、Kontext 关联测试及配置 CLI。
- [ ] 执行可用 lint 和 `git diff --check`，检查新改动不覆盖旧变更。
- [ ] 记录当前环境、测试结果和仍不可运行的历史资产；不将准备好的评测命令标为执行完成。
- [ ] 下一阶段才恢复旧权重/数据并运行表示链条与修复后真实协议。

## 自查

范围不包含训练数学目标重写、生成 replay 或基座切换。loss 与采样权重相互独立；adapter 与 optimizer 的来源明确。兼容默认 proportional 与原 checkpoint resume，协议变化重置选优分数。
