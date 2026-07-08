# Open-Vision-Banana 深度分析计划

## 项目概览
- **名称**: Open-Vision-Banana
- **定位**: 基于 FLUX.1 [schnell] 的通用视觉理解开源实现（Vision Banana 论文复现）
- **版本**: 0.4.2
- **核心思想**: 将视觉任务输出参数化为 RGB 图像，通过图像生成实现视觉理解
- **代码规模**: ~139 个 Python 文件, ~12,320 行源码

## 分析阶段

### Stage 1: 并行模块分析（4个子Agent）

#### Agent 1: 核心算法分析员
- **目标**: 分析 `src/open_vision_banana/core/` 目录
- **文件**: depth_mapper.py, adaptive_depth_mapper.py, normal_mapper.py, segmentation.py, rgb_mapper.py, union_find.py, edge_mapper.py, flow_mapper.py, keypoint_mapper.py, object_detection_mapper.py, depth_edge_mapper.py
- **输出**: 核心算法架构、roundtrip 精度、实现质量、P0/P1/P2 问题

#### Agent 2: 推理与部署分析员
- **目标**: 分析 `src/open_vision_banana/inference/` 目录
- **文件**: pipeline.py, generator.py, prompt.py, decoder.py, optimization.py, checkpoint.py
- **输出**: 推理架构、优化策略、多任务支持、部署就绪度

#### Agent 3: 训练系统分析员
- **目标**: 分析 `src/open_vision_banana/training/` 目录
- **文件**: tuner.py, trainer_interface.py, checkpoint.py, utils.py, logging.py, fsdp_utils.py, deepspeed_utils.py, early_stopping.py, lr_finder.py, optimization.py, latent_cache.py
- **输出**: 训练架构、分布式支持、LoRA 实现、检查点管理

#### Agent 4: 数据与评估分析员
- **目标**: 分析 `src/open_vision_banana/data/` + `src/open_vision_banana/evaluation/` + `tests/`
- **文件**: datasets.py, mixed.py, generation.py, validation.py, evaluator.py, 各 metrics 文件, 全部 tests/
- **输出**: 数据管道、评估指标、测试覆盖、工程实践

### Stage 2: 整合报告
- 合并所有子Agent输出
- 生成分数卡与成熟度评估
- 输出最终 Markdown 报告
