# Open-Vision-Banana 复现、优化与发布设计

状态：Accepted for implementation
日期：2026-07-25
论文：*Image Generators are Generalist Vision Learners*, arXiv:2604.20329v3

## 1. 结论

项目采用两个互不混淆的实验轨道：

1. `paper_reproduction_baseline.yaml`：复现“一个图像生成器通过自然语言和 RGB 输出完成五类视觉任务”的方法，不修改骨干、不增加任务头。
2. `paper_optimized_multitask.yaml`：在同一数据和评测协议上逐项验证数据、采样、LoRA、正则化和吞吐优化。

Nano Banana Pro、原始训练数据、精确超参和训练基础设施均未公开，因此 FLUX.1-schnell 是开放替代基座。未经实际 benchmark 运行，不得把论文表格数字描述为本项目复现结果。

## 2. 事实等级

所有设计、配置和发布材料必须使用以下标签：

| 等级 | 含义 | 本项目示例 |
|---|---|---|
| P：论文明确 | 公式、算法或协议在论文中直接给出 | 深度幂变换；Figure 5 八顶点色路；法线 RGB 映射；实例后处理参数 |
| I：合理推断 | 与论文叙述一致，但数值未公开 | 共享参数、冻结大部分基座、混入生成数据 |
| E：项目扩展 | 为开放复现、效率或质量增加 | FLUX.1-schnell、10% 视觉比例、rsLoRA rank 64、Min-SNR、EMA |

禁止把 I/E 写成“paper default”。现有 9:1、rank 64/alpha 128 是项目基线，不是论文披露值。

## 3. 范围与成功条件

### 3.1 范围

- 五个核心任务：语义分割、实例分割、指代/推理分割、度量深度、表面法线。
- 生成能力保留作为与视觉能力同级的发布门禁。
- 单卡、DDP、FSDP、DeepSpeed 的一致配置契约。
- 从数据清单到 checkpoint、评测 JSON、模型卡的完整可追溯链路。

### 3.2 非目标

- 不声称数值复现闭源 Nano Banana Pro。
- 基线阶段不引入任务专用 head、MoE 或独立任务 adapter。
- 不把训练 loss 当作模型质量的充分证据。
- 不在训练集使用论文 benchmark 的 val/test 样本或其伪标签。

### 3.3 成功条件

- 两个 canonical config 均通过严格配置预检。
- 五任务均能从统一 checkpoint 推理并产生可解码 RGB。
- 每个任务都有固定数据清单、固定 prompt 协议、固定 seed 列表和机器可读指标。
- 优化 checkpoint 同时通过视觉、生成保留、稳定性和吞吐门禁。
- 发布包能够从 config、Git SHA、数据指纹和 checkpoint SHA-256 还原实验。

## 4. 当前状态与已修复缺口

| 区域 | 原状态 | 当前决策/修复 |
|---|---|---|
| 配置解析 | 未知 YAML 字段静默丢弃 | 训练与预检使用 `strict=True`，未知字段立即失败 |
| LoRA | `target_modules`、dropout 未接入 | 已传入 `InstructionTuner`；`null` 时按模型注册表解析 |
| Optimizer | weight decay 固定为 0.01 | 已接入 name、lr、betas、eps、weight decay |
| 训练目标 | 多个 CLI 默认覆盖配置 | Min-SNR、timestep、CFG dropout、flow、EMA warmup 均由配置控制 |
| Seed | CLI 默认 42 覆盖 YAML | CLI 未指定时使用 `training.seed` |
| 数据分支 | `<root>/generation` 被当作视觉任务 | 只进入 generation-preservation 分支 |
| 任务标签 | 多数专用数据集不返回 task | 混合层统一补全并用于色图、增强和 loss 日志 |
| 多任务增强 | 所有样本按 depth 处理 | 每个样本将实际 task 传给 task-aware augmentation |
| 分布式采样 | worker/rank 复制 RNG 序列 | 随机流由 seed、epoch、全局 index 决定 |
| 数据切分 | source split 合并后丢失任务边界 | 训练子集按原数据集返回，保留 task-uniform sampling |
| 可复现记录 | 无统一实验 manifest | 新增严格预检、配置 hash、Git/环境/数据指纹 |

仍需在真实数据上验证：数据语义、全部五任务训练、真实 benchmark、Gemini pairwise judge、显存和吞吐。

## 5. 目标架构

```text
immutable source data
        |
        v
task-specific converters --> validation + dedup + leakage gate
        |                              |
        v                              v
versioned RGB task shards ------> dataset manifest
        |                              |
        +--------- task sampler <------+---- generation-preservation shards
                         |
                         v
 frozen generator + one shared LoRA/rsLoRA adapter
                         |
                         v
 checkpoint + config + git/env/data manifest
                         |
             +-----------+-----------+
             |                       |
       five-task eval          generation eval
             |                       |
             +-------- promotion gate+
                         |
                  release artifact
```

基线维持一个共享 adapter。只有在共享 adapter 的容量瓶颈被三次独立运行和梯度冲突诊断确认后，才允许评估 task-conditioned adapter；该实验必须留在优化轨道。

## 6. 数据方案

### 6.1 通用规则

每个转换后的样本至少记录：

- `sample_id`、`source_id`、任务、数据集、原始 split、许可证和来源 URL/version。
- 输入和目标相对路径、尺寸、内容 SHA-256、转换器 Git SHA。
- prompt 模板 ID、颜色表/色图 ID、坐标系、单位和有效区域比例。
- train/val/test 分配及 group key。scene、video、source image 必须作为 group，不能逐帧随机切分。

预处理门禁：

1. 内容 SHA-256 + pHash 去重；跨任务的同源图像共用 `source_id`。
2. benchmark val/test hash 与训练清单做集合差，交集必须为零。
3. 损坏文件、空 mask、极小有效区域、NaN/Inf、尺寸不匹配必须失败。
4. 原始标注以无损格式保留；训练 RGB 只作为派生物，不能成为唯一真值。
5. 数据版本不可原地覆盖，使用 `dataset_name/version/manifest.jsonl`。

### 6.2 语义分割

- 保留整数 label map 和 ignore/void mask，RGB palette 使用 PNG，不使用 JPEG。
- 每个样本保存完整 class-to-color mapping；评测 Cityscapes 时 prompt 包含全部 19 类，包括当前图中不存在的类。
- 训练 prompt 均衡覆盖自然语言、JSON、RGB tuple、hex 四种形式；颜色分配避免近色。
- 几何增强对图像和 label 同步；label/派生 RGB 使用 nearest 插值。
- 检查 RGB 解码回 label 后像素一致率，目标为 100%（ignore 区域除外）。

### 6.3 实例分割

- 每条指令只请求一个名词类别，每个实例使用唯一且高间隔的颜色。
- 保留 instance ID map、category ID、crowd/ignore 区域，RGB 仅为训练目标。
- 小实例采样需要分桶，避免面积分布被大实例支配。
- 验证 RGB 经五阶段后处理后，实例数量和连通域没有系统性坍缩。

### 6.4 指代与推理分割

- 区分 referring、reasoning、multilingual 三个 prompt 子集并分别报告。
- 表达式覆盖外观、空间、动作、物体关系和非常规用途；不能泄漏 mask/bbox 坐标。
- 同一图像的所有表达式必须进入同一 split。
- 输出二值 RGB mask；空目标必须有显式协议，不能静默当作全黑正常样本。

### 6.5 度量深度

- 主训练源优先使用带真实尺度的合成数据；统一为米，保留 `float32 depth` 和 valid mask。
- 深度缩放/裁剪必须同时更新数值，不能对度量深度做逐图 median scaling。
- P 级编码：`lambda=-3, c=10/3`；Figure 5 路径为 black, red, yellow, green, cyan, blue, magenta, white。
- baseline 只用 `cube_edge`；optimized 才加入 plasma/inferno/viridis/grayscale。
- 对每个数据版本执行 depth -> RGB -> depth roundtrip，分别报告近/中/远距离误差和 P99。
- 相机内参只用于可选 3D 重建评测，不是深度预测输入。

### 6.6 表面法线

- 统一到 camera-space 右手系：+x 右、+y 上、+z 朝相机。
- 转换后验证有效像素范数 `|n|-1` 的 P50/P95/P99；无效区域独立 mask。
- P 级映射：`R=(1-x)/2, G=(1+y)/2, B=(1+z)/2`。
- 水平翻转时取反 x 分量；旋转/透视增强需要完整向量变换，否则禁止。
- 连续法线 resize 使用 bilinear 后重新归一化；发布前增加该专用转换器测试。

### 6.7 生成能力保留集

- 固定一套有许可证的 prompt/image reference 清单，覆盖人物、文字、物体、场景、风格和长尾概念。
- 与视觉训练图像严格去重。
- 固定 inference steps、guidance、strength 和 seeds；同时保存 base 与 tuned 输出。
- 视觉/生成比例是实验变量：建议测试 5/95、10/90、20/80，不能预设 9:1 为论文真值。

## 7. 模型与训练

### 7.1 基线

- 基座：FLUX.1-schnell，冻结 VAE、text encoders 和 backbone 原权重。
- 一个共享 rsLoRA；rank 64/alpha 128 是 E 级项目起点。
- `target_modules: null`，从 model registry 获取 FLUX attention + FFN + context targets。
- 512 分辨率、task-uniform sampling、10% vision、无数据增强、无 EMA/Min-SNR。
- 使用模型原生 flow-matching 目标和 logit-normal timestep。

### 7.2 优化轨道

优先顺序如下，每个实验只改变一组因素：

| ID | 改动 | 候选 | 主要假设 |
|---|---|---|---|
| D1 | 数据门禁 | 基础检查 -> 语义/roundtrip/leakage | 降低标签噪声比加模型容量更有效 |
| S1 | 任务采样 | uniform / proportional / weighted | 防止大数据集支配共享 adapter |
| A1 | task-aware 增强 | off / conservative | 提升泛化且不破坏几何语义 |
| R1 | 数据比例 | 5/95, 10/90, 20/80 | 寻找视觉收益与生成遗忘的 Pareto 点 |
| L1 | LoRA 容量 | rank 32, 64, 128 | 多任务是否需要更高共享容量 |
| L2 | LoRA targets | attention / attention+FFN/context | 识别有效容量位置 |
| O1 | 优化目标 | Min-SNR off/on；gamma 3/5 | 改善 timestep 不均衡 |
| O2 | 正则化 | dropout 0/0.05；EMA off/on | 降低过拟合和 seed 方差 |
| C1 | 深度色图 | cube only / multi-colormap | 增强编码鲁棒性 |
| M1 | 分辨率 | 512 -> 768/1024 curriculum | 提升细边界和法线细节 |

只有 D1、S1、A1、吞吐优化属于低风险候选。任务专用 adapter/head、MoE、额外 decoder 都是高风险架构实验，不能进入复现基线。

### 7.3 Optimizer 与 schedule

- 基线：AdamW，lr `5e-5`，betas `(0.9, 0.999)`，weight decay `0.01`，cosine + 500 warmup steps。
- 优化起点：rank 128 时 lr `2e-5`，betas `(0.9, 0.95)`，warmup 约总 step 的 1%-3%。
- 对 LoRA 参数执行全局 norm clipping 1.0；记录 pre/post clip norm。
- 8-bit AdamW 仅在 CUDA + bitsandbytes 验证后启用，质量门禁与普通 AdamW 相同。
- scheduler 总 step 必须由有效 DataLoader 长度、epoch 和 accumulation 推导；resume 恢复 scheduler/RNG。

### 7.4 效率

- A100/H20：BF16、gradient checkpointing；单卡先测最大稳定 micro-batch，再设置 accumulation。
- 多卡优先 DDP；只有模型/optimizer state 无法放入单卡时使用 FSDP 或 ZeRO-3。
- 记录 images/s、optimizer steps/s、峰值显存、data wait、checkpoint 时间和失败重试。
- latent cache 只有在关闭随机几何增强、色图固定且 cache key 包含 VAE/config hash 时才可启用。
- 数据分片使用本地 SSD cache；所有 rank 读取同一 immutable manifest。

## 8. 评测协议

### 8.1 必报指标

| 任务 | 主指标 | 辅助指标 |
|---|---|---|
| Semantic | mIoU | pixel accuracy、per-class IoU、boundary IoU |
| Instance (SA-Co) | `cgF1` | `pmF1`、`IL_MCC`、AP/AR、失败率 |
| RefCOCOg/ReasonSeg | gIoU、cIoU | Precision@0.5/0.7/0.9、空 mask 率 |
| Metric depth | AbsRel、delta1 | SqRel、RMSE、delta2/delta3、近中远分桶 |
| Surface normal | mean/median angular error | RMSE、11.25/22.5/30 度内比例 |
| Generation | base-vs-tuned pairwise win/tie/loss | CLIPScore、LPIPS、失败率 |
| Efficiency | images/s、peak VRAM | latency P50/P95、artifact size |

`cgF1 = 100 * pmF1 * IL_MCC`。不得改成平方根或均值。

### 8.2 协议固定项

- 每个 checkpoint 使用 seeds `[17, 42, 123]`；发布结果报告 mean、std 和逐 seed 原始 JSON。
- prompt、颜色表、resolution、steps、strength、decoder 参数、dataset manifest hash 全部进入结果文件。
- metric depth 禁止 test-time scale/shift alignment，除非单独标注为非度量协议。
- 实例/指代结果同时报告 RGB 生成失败率，避免只统计成功样本。
- Gemini judge 使用缓存、可恢复 JSONL 和 pair order 随机化；报告精确 pairwise win rate。

### 8.3 晋级门禁

优化候选必须同时满足：

1. 正确性：配置预检、单元测试、RGB roundtrip、数据泄漏检查全部通过。
2. 视觉质量：五任务主指标中至少三项提升；任何一项相对 baseline 退化不得超过 1%。
3. 生成保留：自动指标相对 base 的下降不超过 2%；pairwise 非劣下界不低于 45%。
4. 稳定性：三个 seed 无 NaN/Inf；主指标变异系数不超过 5%。
5. 效率：若质量相当，images/s 至少提升 10% 或峰值显存至少下降 15%。

阈值是 E 级工程门禁，可随真实基线校准，但变更必须新增 ADR，不能回填历史结果。

## 9. 可复现流程

预检：

```bash
PYTHONPATH=src python scripts/validate_training_config.py \
  --config configs/paper_reproduction_baseline.yaml \
  --hash-mode content \
  --output output/manifests/reproduction.json
```

训练：

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/paper_reproduction_baseline.yaml
```

评测示例：

```bash
CHECKPOINT=$(find output/paper-reproduction-baseline -maxdepth 1 \
  -type d -name 'checkpoint-[0-9]*' | sort -V | tail -1)

PYTHONPATH=src python scripts/evaluate_paper_benchmarks.py \
  --benchmark all \
  --model-name black-forest-labs/FLUX.1-schnell \
  --lora-path "$CHECKPOINT" \
  --output-json output/eval/reproduction.json

PYTHONPATH=src python scripts/evaluate_generation_preservation.py \
  --model-name black-forest-labs/FLUX.1-schnell \
  --lora-path "$CHECKPOINT" \
  --dataset-root data/generation_eval \
  --output-json output/eval/generation-reproduction.json
```

每个 run 归档：

- 原始 YAML 与 canonical effective config。
- config/Git/dataset/checkpoint SHA-256。
- `pip freeze` 或锁文件、CUDA/driver/GPU 信息、Docker image digest。
- 训练日志、逐 step LR/loss/grad norm、断点恢复记录。
- 每任务逐样本预测、失败 JSONL、聚合 JSON/CSV 和可视化样本。

## 10. 部署

### 10.1 Artifact

发布目录至少包含：

```text
adapter/
adapter_config.json
training_config.yaml
experiment_manifest.json
evaluation/
MODEL_CARD.md
LICENSES/
SHA256SUMS
```

LoRA 必须声明精确 base model repo + revision、target modules、rank/alpha、PEFT/diffusers 最低版本。当前 ONNX 支持重点是 depth/normal mapper；不能把 mapper export 宣称成完整 FLUX generator ONNX export。

### 10.2 Serving

- 启动时校验 checkpoint、base revision 和 mapper hash；`/healthz` 需要区分进程存活与模型 ready。
- 请求限制 image bytes、像素数、batch、steps 和并发；base64 解码前后都设上限。
- 按 task/resolution/steps 做动态 batching；返回 seed、模型 revision、prompt template ID 和 latency。
- 对 OOM、非有限输出、空 mask、解码失败建立独立错误码和监控。
- 默认不记录原始用户图像；日志仅保留显式允许的匿名统计。
- 生产部署使用 worker queue 和背压，不让同步 FastAPI handler 直接无限并发访问 GPU。

### 10.3 性能目标

在指定硬件 profile 后再冻结绝对 SLO。初始相对目标：

- 同硬件/同采样参数下，optimized P95 latency 不高于 baseline 110%。
- 连续 1,000 请求无显存持续增长，5xx < 0.1%。
- checkpoint 加载失败必须 fail closed，不能回退到未调优 base 且返回成功。

## 11. NFR 与失败模式

| NFR | 目标 | 验证 |
|---|---|---|
| Reproducibility | 同 manifest/seed 产生相同采样序列 | 单 worker、多 worker、DDP 对照测试 |
| Reliability | checkpoint 可原子保存和完整恢复 | 中断/恢复注入测试 |
| Observability | 每任务质量、吞吐、失败率可追踪 | dashboard + machine-readable logs |
| Scalability | 1/8 GPU 结果统计一致 | fixed-step parity run |
| Security | 限制输入和资源消耗 | malformed image、oversize、并发压测 |
| Portability | CPU mapper + CUDA/MPS inference 明确分级 | CI matrix 和模型 smoke test |

主要失败模式及缓解：

| 失败 | 信号 | 缓解 |
|---|---|---|
| 生成遗忘 | pairwise loss、CLIPScore 下滑 | 降低 vision ratio、提高 generation replay、早停 |
| 任务负迁移 | 某任务指标持续下降 | task-uniform/weighted sampling、梯度冲突诊断 |
| RGB 编码漂移 | roundtrip/P99 误差上升 | 无损目标、色表 hash、mapper 单元测试 |
| Benchmark 泄漏 | source/content hash 交集 | 构建阶段硬失败并重建 split |
| 分布式重复采样 | rank 间 sample_id 重复 | index-derived RNG、每 epoch 审计 sample IDs |
| Resume 不一致 | LR/损失跳变 | 恢复 optimizer/scheduler/EMA/RNG 和 manifest |
| OOM/吞吐抖动 | peak VRAM/data wait | micro-batch 探测、本地 cache、queue 背压 |
| 指标幸存者偏差 | 只评成功输出 | 所有失败计入 denominator 并单列失败率 |

## 12. 开源发布清单

- [ ] 论文结果与项目结果分表，并写明 base model 差异。
- [ ] 训练/评测数据卡包含来源、许可证、split、去重和已知偏差。
- [ ] 固定 config、manifest、checkpoint hash 和完整评测 JSON 可下载。
- [ ] 模型卡不含未经本项目运行验证的 SOTA/显存/延迟断言。
- [ ] 一条命令完成 config-only preflight；一条命令完成小数据 smoke test。
- [ ] CI 覆盖 mapper、decoder、metrics、strict config、manifest 和 resume。
- [ ] 发布前运行秘密扫描、依赖许可证扫描和 checkpoint 恶意文件检查。
- [ ] 提供 issue 模板：复现失败必须附 manifest 和硬件信息。
- [ ] 使用 SemVer；模型 artifact 另设 model revision，不覆盖旧权重。

## 13. 分阶段执行

### Phase 0：契约与基线

完成 strict config、有效超参接线、任务/生成分支、确定性采样和 manifest。验收：config-only preflight + 全测试通过。

### Phase 1：数据质量

实现版本化 converter、语义验证、depth/normal roundtrip、source/hash leakage gate。验收：五任务数据报告无 error、benchmark 交集为零。

### Phase 2：真实基线

运行三个 seed 的 baseline，产出五任务、生成保留和效率数据。验收：结果可由 manifest 复跑，不能有缺失任务。

### Phase 3：优化消融

按 D1 -> S1 -> A1 -> R1 -> L1/L2 -> O1/O2 -> C1/M1 顺序实验。验收：只晋级同时通过全部门禁的 checkpoint。

### Phase 4：部署与发布

压测 serving、生成 SHA256SUMS、修订模型卡、做 dry-run Hub 发布。验收：全新环境仅依赖发布包即可运行 smoke inference 和 mapper roundtrip。

## 14. 验收证据边界

代码和配置通过测试只能证明实现契约成立，不能证明论文效果已复现。最终“复现完成”至少需要：真实 checkpoint、固定 benchmark 数据、五任务结果、生成保留结果、三个 seed 方差和公开 artifact hash。当前缺少这些外部训练产物时，项目状态应写为“method-aligned implementation”，而不是“paper result reproduced”。
