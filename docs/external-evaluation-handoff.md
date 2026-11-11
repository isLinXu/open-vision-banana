# 外部评测复现与口径审计（2026-11-11）

公开输入包：[Melo111/open-vision-banana-external-eval](https://huggingface.co/datasets/Melo111/open-vision-banana-external-eval)。该数据集提供评测输入、文件 SHA-256、原始数据获取方法和历史结果快照。它不是训练集，也不是六个完整 benchmark 的统一再授权。

本分支基于上游 main `d3f3426`；现有 I108 服务器作业使用 `62c5b3de8c04e02a18946bf03076e675967d627a`，更早作业部分使用 `01e3d486630db36145a857273f318c2417d62f6b`。同步通过独立 worktree 完成，没有把运行中的作业切到新代码，也没有修改原始看板分数。

## 入口和数据

| 组件 | 入口及职责 |
|---|---|
| 输入恢复/校验 | `scripts/restore_external_eval_data.py`：分卷校验、连接、逐文件安全解包、字节校验；其余数据在上游获取后使用相同工具校验 |
| 调度 | `scripts/run_external_eval_suite.py`：配置驱动，默认仅打印计划；执行时预检→每组 smoke→分片→严格合并，先完成非 RefCOCOg，再运行 RefCOCOg |
| 推理审计 | `scripts/evaluate_external_audited.py`：Qwen 原始输出尺寸检查，query_v2 实际 pipeline prompt 检查 |
| 通用评测 | `scripts/evaluate_paper_benchmarks.py`：数据适配、固定顺序、种子、原生 GT 评分、JSON/CSV |
| GT 适配 | `src/open_vision_banana/data/{datasets,paper_eval_datasets,benchmark_io}.py` |
| 解码与指标 | `src/open_vision_banana/core/*mapper.py`、`evaluation/{evaluator,benchmark,referring,depth}.py` |
| 分片合并 | `scripts/merge_eval_shards.py`、`evaluation/shards.py`：验证连续无重叠区间和相同 metadata，合并原始计数再算指标 |

| 数据 | 集合/条目 | 原生标签 | 共享形式 |
|---|---|---|---|
| NYUv2 | test，654 张 | H5，RGB CHW；depth 480×640，米 | 原始获取说明、精确文件清单/校验值 |
| ETH3D | 本项目适配 split=val，454 张（官方高分辨率训练场景） | 4032×6048，原始 distorted DSLR 图像和对应 float32 深度 | 原始选定素材，CC BY-NC-SA 4.0 |
| DIODE | val/indoors，325 张 | 768×1024，米，depth_mask=1 有效 | 原始三元组，MIT |
| Cityscapes | val，500 张 | 1024×2048，19 个 trainId，ignore=255 | 官方注册下载和转换方法、清单/校验值 |
| ReasonSeg | val，200 张/344 条原始 query | LISA polygon JSON | 官方下载方式、文件清单、query 索引（无文本/多边形副本） |
| RefCOCOg | UMD val，4896 条 expression | COCO2014 原生 polygon/RLE | 官方图像/UMD 标注获取方法、文件清单 |

每 checkpoint 完整一轮 **7173 条**，排除 RefCOCOg 为 **2277 条**。ETH3D 这里的 `val` 是适配器命名，不是官方隐藏测试榜单。不要把 undistorted RGB 与本包原始 depth 配对。文件清单校验全部输入字节；评测 JSON 的 dataset_fingerprint 主要标识路径、顺序和表达式，并不替代逐文件 SHA-256。

## 安装与复现

在合适的隔离环境中安装本项目 `pip install -e '.[inference,evaluation,data,dev]'`；实际 QwenImage21 pipeline 还须使用与模型兼容的 diffusers 版本。保留环境版本，不能仅凭依赖下限声称不同环境结果完全一致。

```bash
hf download Melo111/open-vision-banana-external-eval --repo-type dataset --local-dir /absolute/eval-snapshot
python scripts/restore_external_eval_data.py --snapshot /absolute/eval-snapshot --dataset diode --root /absolute/benchmarks/diode --restore
python scripts/restore_external_eval_data.py --snapshot /absolute/eval-snapshot --dataset eth3d --root /absolute/benchmarks/eth3d --restore
# 其余四项按数据卡取得；root 必须是对应适配器根目录。
python scripts/restore_external_eval_data.py --snapshot /absolute/eval-snapshot --dataset reasonseg --root /absolute/benchmarks/reasonseg
python scripts/run_external_eval_suite.py --config configs/evaluation/external-suite.example.json --output /absolute/new-run
# 替换配置中的路径、revision、SHA256，核验计划后，在空闲 GPU 上执行：
python scripts/run_external_eval_suite.py --config /absolute/suite.json --output /absolute/new-run --gpus 0,1,2,3,4,5,6,7 --execute
```

下载时应固定 HF revision，记录下载返回的 commit。恢复有分卷压缩文件、临时连接归档和原始数据三份空间开销；ETH3D 原始文件约 49.34 GB，DIODE 约 2.55 GB。恢复不会覆盖内容不同的文件。

调度器适用于 Linux，GPU 通过 UUID 映射，每卡单进程，启动时要求占用低于 1000 MiB。使用共享 advisory lock；与既有任务协调时指定相同 `--lock-dir`，例如服务器已有锁所在目录。锁不约束不遵守该协议的外部程序。它不终止其他任务，不自动下载模型，不更换共享环境。失败会停止后续阶段，保留日志；重启同一输出目录仅复用通过验收且配置/源码摘要相同的结果。修复失败输出时应另开目录保存新 attempt。

历史调度另有守护进程、最多三次恢复和前批接续逻辑；新可移植入口不会假装完整迁移这些依赖特定服务器的行为。没有启动新的 GPU 评测来验证本分支。

## 权重和推理协议

I108 起点是 `Melo111/Fusion-C1-KQ0r128-60-40` 的固定 revision `9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d`，不是原始 Qwen。best checkpoint 要检查实际 adapter 的 online/EMA 来源；现有 I108 best 文件的 adapter 来源为 EMA。r64、alpha64、rsLoRA 的缩放是 alpha/sqrt(r)=8。不能把 checkpoint 元数据的 online 字段当成实际 adapter 一定为 online 的证据；不能再次叠加已经 merged 的父 adapter。

配置中的基座 revision 是审计记录，runner 不自动下载/核验基座的 HF 缓存；执行者必须事先验证完整 pipeline 文件。runner 校验显式列出的 adapter 文件 SHA-256，应至少列 adapter_config 和真正装载的 adapter_model，以及必要的 checkpoint metadata。保留完整谱系和基础分数。

默认复现参数：bf16、512（或显式 768）、batch1、50 steps、guidance4、seed42、单种子。Qwen instruction edit 真实传递 `true_cfg_scale=4` 和空格 negative prompt。CLI strength=0.85 会被记录，但该路径不把 strength 传入 Qwen pipeline，不能声称该参数产生了额外的 img2img 作用。主仓新的 depth seed ensemble 是可选诊断，不能混进历史单种子列。

batch1 下每个样本使用 seed42+全局索引，分片起点作种子偏移；改变分片数不会从头重复种子。保持 batch1，避免 batch 索引播种造成批量大小不同的结果不可比。

query_v2 模板：

```text
Identify the object or region referred to by the following query in the image.
Query: {query}
Render its segmentation mask in solid white and all other pixels in black.
```

## 学术口径审计结论

**不能将整个历史看板标记为“官方同口径”。** 公式正确不等于数据划分、query、裁剪、聚合和预测后处理均一致。看板保留原分数，新模式必须另列。

### 深度

预测 RGB 经 cube-edge 最近线段投影得到 t，`metric_power` 逆变换为米；以 lam=-3、c=-10/3 得到 `d=10*((1-t)^(-1/2)-1)`。native 模式先解码，再双线性上采样预测深度到原生 GT；不把 GT 深度通过 RGB 编码回环作为真值。GT 的非有限/非正值和传感器无效像素被排除。

NYU 使用 rows45:471、cols41:601，GT `1e-3<d<=10m`；ETH3D 上限80m；DIODE 当前无上限。NYU/ETH 预测会 clip 至 `[1e-3,cap]`。这些是项目协议，尤其 ETH3D80m、DIODE无上限，不能等同所有论文的标准。应明确论文各自 min/max/crop 后再对比。

AbsRel=`sum(abs(p-g)/g)/N`；RMSE=`sqrt(sum((p-g)^2)/N)`；δ1=`sum(max(p/g,g/p)<1.25)/N`。公式与常用 [Monodepth2 参考实现](https://github.com/nianticlabs/monodepth2/blob/master/evaluate_depth.py) 一致。

但主列是**全局像素加权**；不少论文报告等权图像均值，现有 `_per_image` 列应单独使用。主指标无 GT 尺度对齐，`_aligned` 是 per-image median scaling 诊断，不能代替 metric depth。历史 accumulator 还要求预测 finite 且>0；DIODE 的零值预测可能改变分母；有 cap 时 clamp 又会先改变无效预测行为。该历史政策为复现而保留，不建议宣称其为固定 GT 支持域的严格标准评测。用户已要求保留原分数，本分支没有应用本地未发布的 fixed-GT 实验补丁。

白色附近的深度逆变换敏感，少量极远值会放大 DIODE AbsRel/RMSE；δ1 是阈值比例，两者并非单调对应。不得在看过结果后调 cap、mask 或选择更高分的协议。旧 JSON 只有累加量，没有逐像素预测；无法从中精确重新计算另一口径。

### Cityscapes

19 类 palette 解码后用最近邻上采样标签，原生 GT trainId=255 排除，累计全局 confusion matrix，再算每类 IoU 和19类均值。与 [Cityscapes 官方 class IoU](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/evaluation/evalPixelLevelSemanticLabeling.py) 定义对应，但不是官方 test server 提交结果，也不是 iIoU。新增越界预测报错，避免静默减少有效域；正常19类解码不受影响。官方提交需要 regular labelId，不能直接提交 trainId。若使用 ADE150 palette，须作为跨 palette 诊断，不能列作正常 Cityscapes19 结果。

### RefCOCOg / ReasonSeg

RefCOCOg 严格 UMD val，不回退到 google split；COCO polygon/RLE 用 pycocotools 解码，图像、表达式和非空 mask 要求齐全。预测 RGB 通道均值>0.5 为白色前景，保留所有连通区域，最近邻还原到 GT。cIoU 是所有 query 的交集总和/并集总和；gIoU 是每条 query IoU 的等权均值，不是 box generalized IoU，也不是逐图平均。

本次发现并修复 **ReasonSeg ignore/flag 注释误并入前景**：`--reasonseg-annotation-mode lisa_v1` 使用 OpenCV int32 截断、按光栅面积从大到小覆盖、忽略 flag、ignore=255，评分时同时掩掉 GT 和预测中的 ignore 像素。对实际200张 val 图逐张与 [LISA 官方实现](https://github.com/JIA-Lab-research/LISA/blob/main/utils/data_processing.py) 比对，**0差异**；76张图含 ignore，总计17,910,035个原生像素。核验 SHA/结果见输入包 `audit/`。

通用 CLI 和数据类默认仍为 `legacy_union`，以兼容历史；示例 suite 显式选择 `lisa_v1`。两种模式的 metadata 不同，不能合并分片。历史分数不自动重写。已有结果未保存原始预测，无法仅凭累计 IoU 还原修正分数。

ReasonSeg 这里展开344条原始 query；LISA 和 Vision Banana 的具体 query 选择/重写策略需分别匹配。没有运行 Gemini rewrite。因此“标注处理与 LISA 一致”不代表与论文整套 ReasonSeg 实验完全相同。

更早的 `01e3d486630db36145a857273f318c2417d62f6b` 加载器按图计200条，并对 `text` 列表使用 `str(...)` 生成 query；这些历史分数不是344条集合的简单前200条，也不等于新加载器的 `query_mode=first`。精确复现需固定旧代码并核对实际 prompt。报告可保留跨 query 协议比较，但必须保留样本数和协议差异。

## 验收与共享

验收实际样本数、零 skipped/failed、指标有限、GT 原生尺寸、实际输出画布、权重谱系、SHA、prompt、query/annotation 模式；分片区间无遗漏/重复且 metadata 一致。报告保留原始精度、单位与有效像素，不因为 query 或合理有效域不同自动否定比较，但必须展示这些差异。

本次最终 CPU 回归139项通过，包含恢复、suite 计划、实际 pipeline 审计钩子、ignore mask、深度解码和分片合并测试，详见数据包 `audit/validation.json`。无 GPU 重测，所以不承诺 corrected 模式会提高分数。公开结果快照中尚未完成的 checkpoint 项目仍是进行中，不能当作零分或已完成。

原始数据来源与许可： [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[ETH3D CC BY-NC-SA4](https://www.eth3d.net/)、[DIODE MIT](https://diode-dataset.org/)、[Cityscapes 禁止转发条款](https://www.cityscapes-dataset.com/license/)、[ReasonSeg/LISA](https://github.com/JIA-Lab-research/LISA#dataset)、[RefCOCOg REFER](https://github.com/lichengunc/refer)、[COCO 使用条款](https://cocodataset.org/#termsofuse)。代码许可和数据许可分别适用；数据包没有覆盖上游权利声明。
