# 第一阶段：训练实验契约修复

## 已完成

- 单一配置链路：默认值 < YAML < 显式 CLI；严格字段检查、类型/范围验证、双向布尔参数。
- 字符串任务损失权重正常解析；LoRA targets override 经独立模型配置传入后端，不污染注册表。
- `--print-effective-config` 不创建输出目录、不加载模型；运行时保存配置、任务计数、采样概率和可训练参数名。
- uniform/sqrt 独立采样器，全局确定性有放回抽样后按 rank 分片；默认 proportional 保留原遍历行为。采样权重支持 task alias，与 loss 权重分离。
- 所请求训练任务缺数据、缺文件和空 loader 明确失败；读取失败不再随机替换。数据构建早于模型加载。
- 验证固定每任务样本预算（默认 40），直接构建 metadata 子集以限制读图；按样本累计、按任务宏平均 flow loss，恢复 RNG 和模型模式。
- EMA 开关实际生效；best adapter 导出被验证 EMA，online 权重及 optimizer 独立用于续训。导出异常恢复 online，不更新最佳分数。
- 保存 selection_protocol，配置或 manifest 内容变化时重置旧 best 分数；顶层空 resume 不覆盖 checkpoint.resume，冲突硬失败。
- 显式 warmup_steps=0 保留。

## 验证

Python 3.11.16，项目 `.venv`，CPU。

```bash
PYTHONPATH=src .venv/bin/python -m pytest \
  tests/test_unified_training_config.py tests/test_unified_task_sampling.py \
  tests/test_unified_trainer_validation.py tests/test_unified_trainer_resume.py \
  tests/test_unified_trainer_ddp.py tests/test_ema_checkpoint.py \
  tests/test_model_backend_lora_targets.py tests/test_training_prompts_roundtrip.py \
  tests/test_kontext_experiment.py tests/test_manifest_portability.py \
  tests/test_depth_decode.py tests/test_native_resolution_eval.py \
  tests/test_openvocab_semantic.py tests/test_trivial_baselines.py -q
```

结果：**205 passed**。包含 CPU 两进程梯度同步测试，不代表真实 GPU 训练可用。新增回归先复现故障再修复；复核额外发现并修正验证读图预算、采样别名、manifest 内容指纹和 resume 空值问题。

已实际执行 Kontext 配置 `--print-effective-config`。未下载基础模型，未运行真实推理/训练，未自动提交。保留原有未提交改动。

## 明确限制

- `vision_ratio != 1` 与非空 tar_index 现在明确拒绝，不再假装生成混训/其他数据入口生效。
- 历史另一套 YAML schema 仍不能直接执行，未知字段会报迁移错误；没有自动转换 epochs 或猜测生成数据 manifest。
- 选优仍是 flow loss，不包含辅助损失，不是论文解码指标；正式晋级必须运行解码后基准及结构闸门。
- 采样日志记录 rank 0 的实际 count/ratio，并记录 world_size；不冒充全局观测采样统计。
- 续训仍没有精确恢复 RNG 和数据游标。协议哈希包含 manifest 内容，但没有目标图像内容指纹；同路径替换图像需要单独资产审计。
- EMA warmup 旧字段的数学行为本次未改，不宣称该参数改变了更新公式。
- 尚未执行完整仓库测试套件，也未进行真实 PEFT/量化模型加载验收。

## 历史资产核查

注册表四条 local_path 全部不存在：

- `sana-depth-relative-v1`
- `sana-hypersim-metric-v1`
- `sana-kitti-metric-v1`
- `sana-1600m-multitask-v1`

当前 `data/` 不存在。当前挂载卷未见历史训练数据盘。测试/入口检查生成的 output 不代表历史资产恢复。

因此旧 checkpoint 重评、真实 GT → RGB → VAE → 解码链条及结构闸门**尚未执行**。下一步需要提供历史训练机可访问路径或恢复位置、实际 checkpoint、修复后 benchmark GT 与对应 VAE 权重。不能用随机初始化 VAE 或合成输入替代这一步并声称真实链条通过。

## 配置检查命令

```bash
PYTHONPATH=src .venv/bin/python scripts/train.py \
  --config configs/experiments/flux_kontext_depth_gate.yaml \
  --print-effective-config
```

此命令只验证有效参数，不验证文件存在或真实模型训练。资产恢复后先以同一权重重跑修复后协议，再决定单任务训练及共享模型混训。
