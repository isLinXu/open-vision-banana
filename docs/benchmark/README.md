# Benchmark 使用与扩展

本入口基于 main [`b490e2d3c91d5909de1ef9c9416768ec364ab884`](https://github.com/isLinXu/open-vision-banana/tree/b490e2d3c91d5909de1ef9c9416768ec364ab884)，统一编排模型、尺寸和任务。候选依据见 [checkpoint-selection.md](checkpoint-selection.md)，已执行结果见 [testing.md](testing.md)。随机 Tiny 验证组件；LR-ASPP 是独立预训练参考。

完整项目要求 Python ≥3.10、PyTorch ≥2.0。以下命令从仓库根目录执行，`python` 指向所选环境；数据、权重及结果均使用可替换的相对路径。

## 快速运行

先列矩阵，不导入 torch、不加载模型或访问权重：

```sh
python -m scripts.benchmark --config scripts/benchmark/configs/laptop-smoke.json --dataset-root ./data/flickr30k-images --plan
```

CPU 合成 smoke 无需下载；真实图片 GPU 测试使用已备好的本地权重：

```sh
python -m scripts.benchmark --config scripts/benchmark/configs/synthetic-smoke.json --device cpu --output ./benchmark-output/synthetic
python -m scripts.benchmark --config scripts/benchmark/configs/laptop-smoke.json --dataset-root ./data/flickr30k-images --checkpoint lraspp-voc21=./weights/lraspp_mobilenet_v3_large-d234d4ea.pth --device cuda --output ./benchmark-output/flickr-gpu
```

默认禁止下载；只有 `--allow-download` 才允许获取权重。`--checkpoint MODEL_ID=PATH` 可重复，未知 ID 报错。输出目录必须为空；OOM 或不支持的任务不会静默更换模型、设备或尺寸。

旧 Python／PyTorch 只可用 `--allow-legacy` 做 Tiny／torchvision 无 GT 组件测试，不能启用完整扩散流水线。旧加载器缺 `weights_only=True` 时，可信本地 `.pth` 还需复制 profile、保留其他 options，并显式设置对应模型 `options.trusted_checkpoint=true`；默认仍为 false。优先使用现代环境或真实 safetensors，改后缀不改变容器格式。

## 配置与路径

JSON `schema_version=1`；未知字段、重复 ID、非法尺寸与后端不支持的 options 均失败。模型、尺寸与任务独立配置：

| 部分 | 主要字段 |
|---|---|
| models | id、backend、checkpoint、base_model_id、revision、weights_variant、options |
| sizes | id、max_edge、alignment |
| tasks | semantic／depth／normal，或 name 与 options 对象 |
| dataset | kind、root／manifest、include、max_samples、label_space、ignore_index、depth_units、depth_protocol |
| timing／output | warmup、repeats、cpu_threads；visualizations、raw_predictions |

配置中的本地路径相对配置文件目录，CLI 路径相对当前工作目录；manifest 内容相对 manifest。本地 base_model_id 使用 `./` 前缀，与远程模型 ID 区分。results.json 使用 `dataset/`、`checkpoints/<model-id>/`、`base-models/<model-id>/` 等逻辑别名，不能当成重跑 profile；复现仍用原 profile 与 CLI roots。prepared-checkpoint.json 的 adapter／source 路径相对其缓存根目录，移动完整缓存后可解析。

长边缩放到 max_edge，再向下对齐 alignment；不 padding／crop，小图会放大，宽高比可能略变，实际尺寸写入结果。laptop profile 使用 64／128／256、alignment 8；扩散模型需按 VAE 约束另定。LR-ASPP 的这种缩放不同于官方短边 520px 协议。

## GT 与指标

`dataset.kind=images` 只保存定性预测和成本；无像素 GT 的 Flickr30k 不计算 mIoU／准确率。正式质量评测用 UTF-8 JSONL manifest，例如：

```jsonl
{"id":"sample001","image":"images/sample001.jpg","gt":{"semantic":"labels/sample001.png"},"valid_masks":{"semantic":"masks/sample001.png"}}
```

| 任务 | GT 与契约 |
|---|---|
| semantic | 整数 H×W PNG／NPY；label_space、ID 顺序、ignore_index 必须一致，RGB 图不是 label GT |
| depth | 标量 H×W NPY；depth_units=meters／normalized 与解码 representation 一致 |
| normal | 浮点 H×W×3 NPY；明确相机坐标、轴向、单位向量与 valid mask |

valid_masks 为同 GT 网格的 PNG／NPY，非零有效。LR-ASPP 为 voc21，默认 Tiny 为 ade20k150；跨标签空间需先定义共同映射。语义以全局混淆矩阵累计；指标在原始 GT 网格计算，labels 用 nearest，连续值用 bilinear，法线再归一化。深度需显式 colormap，metric_power 还需 lam/c；NYU／KITTI／ETH3D 协议边界见测试记录。

## 训练权重

独立准备固定 revision 的 LoRA，仅此命令显式允许下载；不获取基座：

```sh
python -m scripts.benchmark.checkpoints --spec scripts/benchmark/configs/checkpoint-candidates.json --candidate fix_best3_s1 --output ./benchmark-output/checkpoint-cache --allow-download
```

准备器校验来源、EMA／online、张量及 hash，生成 adapter 与 prepared-checkpoint.json；缓存齐全后去掉下载选项可重复运行。EMA 后端还检查 provenance 和来源张量，不能用在线 adapter 改名冒充 EMA。

trained-sana.json 是默认拒绝执行的模板。先核对服务器源码／patch、基座 revision、timestep／velocity、VAE 缩放、准确 prompts、深度表示、法线 convention 和 palette，再复制配置填写本地基座、adapter 与 provenance。完成核验后才设置 inference_contract_verified=true；该字段本身不是兼容证明。6GB 显存下的完整 SANA／卸载尚未测试。

## 扩展与产物

流程为 profile → 样本／矩阵 → backend → Prediction → task decode／可选 GT 累计器 → 产物。[backends.py](../../scripts/benchmark/backends.py) 注册 tiny、torchvision_lraspp、vision_banana；输入为设备上的 [1,3,H,W] RGB [0,1]，返回 RGB 编码或 labels，并声明任务与契约。

- 新 checkpoint／尺寸：添加 profile，复用 backend，不复制 runner。
- 新模型：register_backend，实现 prepare／predict／describe／close，隔离 normalization 和依赖。
- 新任务：在 [tasks.py](../../scripts/benchmark/tasks.py) 注册 decode 与指标，复用主分支 core／evaluation；补 GT、mask、边界反例和聚合不变性检查。
- 协议变化：版本化 schema／palette／resize／ignore 策略，保存权重、源码与输入 hash。

结果入口为 results.json；逐样本位于 `artifacts/<model>/<size>/<task>/<sample>/`，含 prediction.npy、prediction-rgb.png、overlay.png、preview.png，语义另有 labels.png。保存解析配置、实际尺寸、运行版本、来源 hash 与失败原因；原始预测网格与 GT 指标网格分别记录。

计时包含模型准备、同步后的 backend.predict，以及最后一次预测的 decode 和配对延迟；预测与配对统计样本数不同。resize、指标、I/O 不计入延迟，尚无 host RAM／端到端测量。比较须固定硬件、dtype、图片、步数和卸载策略；allocated／reserved 不等于含驱动的总显存。
