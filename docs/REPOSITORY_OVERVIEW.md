# Open Vision Banana 仓库概览

> 文档状态：仓库结构与入口的工程概览
>
> 盘点日期：2026-08-25
>
> 说明：本文档基于当前工作树的静态阅读整理而成。编写时未运行测试、构建或训练命令；运行前请以代码和配置的实际状态为准。

## 1. 项目定位

Open Vision Banana 是一个以扩散模型为基础的通用视觉学习框架。项目将输入图像、任务提示和生成模型结合起来，把模型输出的 RGB 结果解码为多种视觉任务结果，同时提供训练、评测、部署和模型导出能力。

当前项目同时包含研究复现代码、实验脚本和工程化模块，因此仓库规模较大，部分目录仍保留历史实验与阶段性产物。

## 2. 顶层结构

```text
.
├── src/open_vision_banana/   # 核心 Python 包
│   ├── core/                 # 深度、法线、分割、边缘、光流、关键点等解码器
│   ├── data/                 # 数据集适配、混合采样、生成数据与校验
│   ├── inference/            # 模型加载、Prompt、LoRA、生成与解码
│   ├── training/             # Flow Matching、PEFT、EMA、检查点及分布式训练
│   ├── evaluation/           # 视觉任务指标与评测流程
│   ├── serving/              # FastAPI 服务适配层
│   └── export/               # ONNX / TorchScript 导出
├── scripts/                  # 训练、推理、评测、数据准备和实验脚本
├── configs/                  # 训练、硬件、DeepSpeed、实验及检查点配置
├── tests/                    # 单元、集成、GPU、数据和基准测试
├── docs/                     # 快速开始、复现报告、ADR 和实验记录
├── docker/                   # Dockerfile、Compose 和容器说明
└── .github/workflows/        # CI/CD 工作流
```

核心包使用 `src` 布局，包的版本和构建配置位于 [`pyproject.toml`](../pyproject.toml)。

## 3. 技术栈

| 类别 | 组件 |
| --- | --- |
| 语言与构建 | Python 3.10+、Setuptools、`python -m build` |
| 深度学习 | PyTorch、TorchVision |
| 扩散模型 | Diffusers、Transformers、Accelerate |
| 参数高效训练 | PEFT、LoRA / rsLoRA |
| 数据与图像 | PyYAML、Pillow、可选 NumPy / HDF5 / COCO API |
| 服务与导出 | FastAPI、Uvicorn、ONNX、ONNX Runtime |
| 质量保障 | pytest、Ruff、mypy、GitHub Actions |

基础依赖和可选依赖分别定义在 [`pyproject.toml`](../pyproject.toml) 与 [`requirements.txt`](../requirements.txt) 中。

## 4. 主要入口

| 场景 | 入口 | 说明 |
| --- | --- | --- |
| 单图或批量推理 | [`scripts/inference.py`](../scripts/inference.py) | 支持 depth、normal、semantic、instance、referring、edge、flow、keypoint、object detection 和 depth-edge 等任务 |
| 模型训练 | [`scripts/train.py`](../scripts/train.py) | 从 YAML 配置加载模型、数据、优化器和训练策略 |
| 模型评测 | `scripts/evaluate.py`、`scripts/evaluate_model.py`、`scripts/evaluate_paper_benchmarks.py` | 分别覆盖通用评测和论文基准评测 |
| 统一实验启动 | [`scripts/unified_launch.sh`](../scripts/unified_launch.sh) | 汇总常用训练、推理或评测启动逻辑 |
| HTTP 服务 | [`src/open_vision_banana/serving/fastapi_app.py`](../src/open_vision_banana/serving/fastapi_app.py) | 提供 `create_app` 应用工厂 |
| 模型注册 | [`src/open_vision_banana/model_registry.py`](../src/open_vision_banana/model_registry.py) | 管理模型、LoRA 和检查点别名 |

## 5. 核心数据流

### 5.1 推理链路

```text
输入图像
  → 归一化到 [-1, 1]
  → PromptBuilder 构造任务指令
  → BaseGenerator / Diffusers / LoRA 生成 RGB 结果
  → RGBMapper / VisualizationDecoder
  → 深度、法线、分割、边缘、光流、关键点或检测结果
```

推理流程的主要实现位于 [`inference/pipeline.py`](../src/open_vision_banana/inference/pipeline.py)、[`inference/generator.py`](../src/open_vision_banana/inference/generator.py) 和 `core/` 下的各种 mapper / decoder。

### 5.2 训练链路

```text
数据集样本
  → prompt + input image + target RGB
  → MixedVisionDataset 混合视觉任务与生成数据
  → VAE latent 编码
  → InstructionTuner / Flow Matching
  → PEFT rsLoRA、优化器、EMA
  → checkpoint、指标和实验元数据
```

训练主流程由 [`scripts/train.py`](../scripts/train.py) 驱动，训练核心位于 [`training/tuner.py`](../src/open_vision_banana/training/tuner.py)。

## 6. 安装与常用命令

### 6.1 安装开发依赖

```bash
pip install -e ".[dev]"
```

按使用场景可追加 `inference`、`training`、`evaluation`、`data`、`serving` 或 `export` extras；完整组合为 `all`。

### 6.2 推理

```bash
python scripts/inference.py \
  --image input.jpg \
  --task depth \
  --output-dir output/
```

使用 LoRA 时，可通过 `--lora-path` 指定本地路径、Hugging Face 标识或注册表别名。实际权重是否可用，需要先检查 [`configs/checkpoint_registry.yaml`](../configs/checkpoint_registry.yaml)。

### 6.3 训练

```bash
PYTHONPATH=src python scripts/train.py \
  --config configs/paper_reproduction_baseline.yaml
```

训练前应确认数据路径、模型下载权限、显存/设备配置和 checkpoint 配置；分布式训练还需要检查 DeepSpeed、FSDP 或 DDP 相关设置。

### 6.4 质量检查与构建

```bash
python -m pytest tests/ -v
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
mypy src/
python -m build
```

CI 工作流位于 [`.github/workflows/ci.yml`](../.github/workflows/ci.yml)，覆盖多个 Python 版本、静态检查、端到端测试、训练 smoke test、Docker 构建和性能回归。

## 7. 当前已知问题与风险

以下项目来自本次静态盘点，尚未在本次整理中修复：

1. **工作树非干净状态。** 当前存在较多已修改文件和新增实验文件。后续修改前应先区分已有用户变更与新文档变更。
2. **检查点可能不可直接使用。** `configs/checkpoint_registry.yaml` 中部分官方 `vision-banana-*` 条目仍为占位配置，`registry:vision-banana` 可能无法解析到已发布权重。
3. **文档入口疑似过期。** README 中仍出现 `scripts/validate_data.py`，当前仓库中实际相关脚本为 `scripts/validate_dataset.py`。
4. **Docker 开发目标需核对。** `docker/docker-compose.yml` 使用 `target: dev`，但需要确认 [`docker/Dockerfile`](../docker/Dockerfile) 是否声明了对应的多阶段构建目标。
5. **依赖来源存在漂移风险。** `requirements.txt` 与 `pyproject.toml` 并非完全一致；源码使用的依赖应统一纳入明确的安装入口。
6. **配置注释可能滞后。** `configs/training_config.yaml` 对 `tasks` 的说明需要与 `scripts/train.py` 当前读取 `cfg.data.tasks` 的行为重新核对。
7. **论文复现仍有边界。** [`docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md`](./OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md) 记录了 benchmark、metric depth、normal target 和 semantic segmentation 等待补齐或校准的事项。内部相对深度指标不能直接当作论文中的 δ1 / AbsRel。

## 8. 建议的后续核验顺序

1. 在保留现有工作树改动的前提下，运行最小测试集和静态检查。
2. 核对 `pyproject.toml`、`requirements.txt` 与实际 import 的依赖差异。
3. 验证 checkpoint registry 中至少一个可用权重的下载、加载和最小推理闭环。
4. 验证 Docker Compose 的开发构建目标。
5. 修正 README、配置注释和脚本名称之间的文档漂移。
6. 将论文复现指标与内部诊断指标明确分开，并为缺口补充可重复的评测命令。

## 9. 相关文档

- [`docs/quick-start.md`](./quick-start.md)：快速开始
- [`docs/PAPER_REPRODUCTION_QUICKSTART.md`](./PAPER_REPRODUCTION_QUICKSTART.md)：论文复现快速入口
- [`docs/OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md`](./OPEN_VISION_BANANA_DEEP_ANALYSIS_v130.md)：深度分析与复现缺口
- [`CONTRIBUTING.md`](../CONTRIBUTING.md)：贡献与质量检查约定
- [`docker/README.md`](../docker/README.md)：容器化说明
