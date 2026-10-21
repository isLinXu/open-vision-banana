# Open Vision Banana Code Wiki

> 源码盘点日期：2026-10-16  
> 包版本：`0.5.0`，来自 `src/open_vision_banana/__init__.py`。  
> 基准：本地当前工作树，包含已有未提交源码和实验文件；不是某次发布的功能保证。

## 文档导航

> 本 Wiki 目前只完成第 1 章；第 2–9 章为规划中的目录，尚未撰写。

1. [整体架构与仓库地图](01-architecture.md)：项目定位、技术路线、分层、依赖方向、端到端调用链。
2. 核心算法与任务编码（待补）：深度、法向、分割、边缘、光流、关键点和检测。
3. 数据系统与数据契约（待补）：数据集适配、manifest、采样、归档读取、数据准备。
4. 训练系统（待补）：统一训练器、模型后端、Flow Matching、编辑模型、EMA、恢复、旧版设施。
5. 推理与模型管理（待补）：公共 API、模型注册、权重解析、采样分支、提示和分块推理。
6. 评估与实验协议（待补）：指标、累加器、原生 GT、编码契约、论文基准及能力边界。
7. 安装运行与部署（待补）：环境依赖、CPU 检查、数据准备、训练、推理、评估、服务和导出。
8. 测试维护与扩展指南（待补）：测试地图、CI、扩展位置、已知边界、文档核验记录。
9. 完整源码 API 索引（待补）：全部包内 Python 文件、定义行号、类与函数签名、源码说明和内部导入。

推荐首次阅读顺序：架构 → 数据 → 训练 → 推理 → 运行。只接入模型时先读推理与运行；新增任务时先读核心算法、数据和扩展指南。

## 项目一句话

这是一个研究型 Python 框架：把视觉任务的目标编码成 RGB 图像，以图像生成模型配合 LoRA 学习任务指令，再把生成的 RGB 解码为结构化视觉结果。它同时包含确定性深度探针和独立 Tiny CNN 对照，不能把所有实验路线都等同于生成式论文复现。

## 阅读约定

- 本套文档解释实际代码，旧报告用于补充背景，不覆盖当前实现。
- 专题章节解释关键调用链；API 索引提供完整文件级覆盖，不意味着每个辅助函数都经过运行验证。
- 相对源码链接指向当前仓库，`#L...` 表示定义位置；源码变动后行号可能漂移。
- 安装、训练命令中的 `/absolute/path/...` 是需要替换的数据或权重路径，不是仓库自带资源。
- 包内 `core/` 的算法不需要下载模型，但仍依赖 PyTorch，部分算法涉及 NumPy 或可选 SciPy。
- “支持任务”指编码、解码或调用接口存在；不代表权重可用、任务质量达标或论文指标复现。

## 最重要的工程边界

1. `scripts/train.py` 使用 `training.config.TrainingConfig` 和 `Trainer`，不是旧版 `VisionBananaConfig` / `InstructionTuner`。
2. 统一训练器只接受 `pair_manifest`；`data.tar_index` 配置字段仍被拒绝，但 `ManifestVisionDataset` 会自动发现 `data_dir` 或 manifest 旁的 `tar_index.json` 用于随机读取。`vision_ratio < 1.0` 仅在 `tasks` 含 `generation` 时可用（按 `1 - vision_ratio` 混入生成保留样本）。
3. 统一训练器采用单进程或 `torchrun` 下的 LoRA 梯度同步，不接受 `--fsdp` / `--deepspeed`；旧版分布式类不能证明该入口支持这些后端。
4. 深度色图、深度单位、语义调色板和任务提示必须匹配训练数据。全局默认编码契约不是所有 checkpoint 的通用契约。
5. `registry:vision-banana` 等条目仍为占位；注册文件中的 `available` / `local` 也不等于本次已经验证远端或本机权重。
6. `export_full_pipeline_onnx()` 实际只导出 FLUX transformer，是已弃用别名；不是完整生成服务的 ONNX 导出。

## 资料入口

- [项目 README](../../README.md) 与 [包元数据](../../pyproject.toml)。
- [快速开始](../quick-start.md)、[论文复现快速入口](../PAPER_REPRODUCTION_QUICKSTART.md)。
- [历史路线边界](../ROUTE_BOUNDARIES.md)、[架构决策记录](../adr/)。
- [近期实验记录](../EXPERIMENTAL_RESULTS_2026-10-10.md)、[消融基线](../ABLATION_BASELINE.md)。

本次只新增 Wiki 文档，不修改训练、推理、数据和测试实现。
