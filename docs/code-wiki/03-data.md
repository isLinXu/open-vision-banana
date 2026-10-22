# 数据系统与数据契约

[返回首页](README.md)

> 盘点基准：`main` 在 PR #37–#40 合入之后。本章解释"公开数据 → manifest → 训练 batch"的真实代码路径，以及每一步强制执行的契约。命令以仓库根目录为工作目录，`PYTHONPATH=src`。

## 1. 一条链路，五个阶段

```text
configs/datasets/sources.tsv ──► scripts/download_datasets.sh ──► data/raw/<source>/
        (注册表)                      (ms / hf / url / manual)          (下载后的原始布局)
                                                                          │
   ┌──────────────────────────────────────────────────────────────────────┘
   ▼  每个源一个准备脚本，输出 manifest 相对路径的 pairs.jsonl
scripts/prepare_hypersim.py / prepare_tartanair.py / build_openvocab_semantic_manifest.py /
build_instance_manifest.py / build_res_manifest.py / build_generation_replay_manifest.py
   │
   ▼
scripts/merge_multitask_manifest.py      路径重定位、按任务 cap、补 val
   │
   ▼
scripts/build_qwen_baseline_manifest.py  零样本审计（路径 + COCO image-id）、米制深度过滤
   │
   ▼
training.trainer.ManifestVisionDataset   每任务一个实例 → ConcatDataset → TaskWeightedSampler → _collate_fn
```

两个测试把整条链钉死：[`tests/test_pipeline_smoke_all_tasks.py`](../../tests/test_pipeline_smoke_all_tasks.py)（fixture 走到六任务 batch）和 [`tests/test_trainer_system_all_tasks.py`](../../tests/test_trainer_system_all_tasks.py)（真实 `Trainer` + CPU 假后端）。新增数据源必须让两者通过。

## 2. 注册表与下载器

[`configs/datasets/sources.tsv`](../../configs/datasets/sources.tsv) 每行一个源：`name / group / tasks / source / locator / include / dest / post / size / license / note`。

- `group`：`train`、`replay`（生成保留数据）、`eval`（论文基准）。`--group train` **永远不会**拉 `eval` 行；`tests/test_download_datasets_script.py` 把全部 16 个论文基准钉在 `eval`。
- `source`：`ms`（ModelScope CLI，缺 CLI 时退到 Python `snapshot_download`）、`hf`（`hf` / `huggingface-cli`，默认 hf-mirror）、`url`（curl 断点续传 + 失速中止）、`manual`（登录墙，打印说明后跳过）。
- `include`：按文件选下（`HYPERSIM_SCENES`、`TARTANAIR_CHUNKS`、`OMNIEDIT_SHARDS` 环境变量展开）。
- `post`：`unzip | untar | *_each | un7z_each | coco2014_links | refer_umd | refer_unc | nyu_mat | genai_prompts`，可用 `+` 串联；解压后写 `.extracted` 标记保证幂等。

`post` 步骤里有三个真正的格式转换器，它们存在是因为上游格式和本仓库 Dataset 的期望不一致：

| post | 脚本 | 为什么需要 |
|:--|:--|:--|
| `refer_umd` / `refer_unc` | [`scripts/convert_refer_parquet.py`](../../scripts/convert_refer_parquet.py) | UNC 原始 REFER 主机已失效；ModelScope parquet 镜像带完整多边形 / 句子 / split，重建成 `instances.json + refs(*).p` 后下游零改动 |
| `nyu_mat` | [`scripts/convert_eval_sets.py nyu-mat`](../../scripts/convert_eval_sets.py) | `nyu_depth_v2_labeled.mat` 是 MATLAB v7.3 列主序；转成 `test/images + test/depths`（mm PNG），用 `splits.mat` 得到 Eigen 654 |
| `genai_prompts` | 同上 `genai-bench` | parquet → `prompts.jsonl` |
| `coco2014_links` | 下载器内联 | RefCOCO 用 2014 文件名，COCO 2017 图像用**硬链接**映射（`build_res_manifest` 的路径穿越防护拒绝软链） |

## 3. 准备脚本与每任务契约

所有脚本输出 **manifest 相对路径**（`data/manifest.resolve_manifest_path` 约定），行字段至少 `task / input / target / prompt / split`。

| 任务 | 脚本 | target 契约 | 行上的额外字段 |
|:--|:--|:--|:--|
| depth | `prepare_hypersim.py`、`prepare_tartanair.py` | 米制 → 幂变换（λ=−3, c=10/3）→ cube-edge RGB；Hypersim `depth_meters` 是**射线距离**，`--ray-distance auto` 默认转平面 z（[`data/depth_geometry.py`](../../src/open_vision_banana/data/depth_geometry.py)） | `target_kind`，Hypersim 还写 `metric_depth/*.npy` 供审计 |
| normal | `prepare_hypersim.py --tasks normal` | `normal_cam` 直接过 `NormalRGBMapper`（Hypersim 相机系 = 论文 +x右/+y上/+z朝外）；背向比例 >10% 告警 | `target_kind: hypersim_normal_cam_rgb` |
| semantic | `build_openvocab_semantic_manifest.py`（ADE20K / COCO-Stuff）、`prepare_hypersim.py --tasks semantic` | Fig. 2 开放词表：每行随机抽类别→颜色映射和 prompt 风格（`json_named / json_tuple / natural / hex`），未标注像素取 prompt 的 `background` 色 | `mapping`（精确图例，训练期优先于解析 prompt）、`style` |
| instance | `build_instance_manifest.py`（COCO / LVIS）、`prepare_hypersim.py --tasks instance` | 黑底，每实例一色，颜色两两 ≥60（[`data/instance_palette.py`](../../src/open_vision_banana/data/instance_palette.py)），Fig. 3 三种句式；剔除 `iscrowd` 与 `not_exhaustive` | `category`、`num_instances`、`background_color` |
| referring | `build_res_manifest.py` | `binary_white_black_v1`：`Segment {expression} in solid white. Render all other pixels black.`（[`referring_contract.py`](../../src/open_vision_banana/referring_contract.py)） | `expression`、`encoding`、来源 / 协议 / split |
| generation | `build_generation_replay_manifest.py` | T2I：`input == target`；编辑：`(源图, 指令, 结果)` 三元组。支持 JSONL / CSV / parquet（HF `Image` bytes 落盘）/ WebDataset tar / 图目录 + caption 文件；`--preset omniedit` | 无；编辑底座训练时 `Trainer` 拒绝恒等行 |

## 4. 合并与审计

[`merge_multitask_manifest.py`](../../scripts/merge_multitask_manifest.py)：对每个输入 manifest 按**其自身目录**解析路径（历史 bug：曾按 CWD 解析，所有相对路径行被静默丢弃），重写为相对输出文件（或 `--absolute`），`--cap` 每任务上限，`--val-percent` 给没有 val 的任务分 val（训练器要求"要么全有 val 要么全没有"）。产物 `pairs.audit.json`：`per_task_dropped` 必须为 `{}`。

[`build_qwen_baseline_manifest.py`](../../scripts/build_qwen_baseline_manifest.py)：
- 零样本：`ZERO_SHOT_BENCHMARK_PATTERNS` 路径匹配 **加** `--exclude-image-ids refs(umd).p` 的 COCO image-id 匹配（RefCOCOg UMD val/test 的图像就在 COCO train2014 里，光靠路径拦不住）。
- 米制深度：`is_valid_metric_cube_edge_target` 回环检查，相对深度 / 灰度 / 非 cube-edge 一律剔除。
- `--min-cityscapes-share` 用于旧实验；公开数据管线传 `0`。
- 产物 `pairs.audit.json` 记录各类 rejected 计数和每任务 train/val 数。

开训前最后一道闸：`data.validation.validate_manifest_contract(manifest, model_name=..., data_root=...)`，检查 LoRA 目标是否含注意力、语义图例是否会被文本编码器截断、深度 prompt 与 target 是否矛盾、`uniform_task_prompts` 冲突等。

## 5. 训练期数据加载（`ManifestVisionDataset`）

定义在 [`training/trainer.py`](../../src/open_vision_banana/training/trainer.py)。要点：

- 每个任务一个实例（`tasks={task}`），`TASK_ALIASES` 把 `seg → semantic`、`metric_depth → depth`。`Trainer._build_datasets` 把它们 `ConcatDataset`，再用 `TaskWeightedSampler`（`proportional | uniform | sqrt`，以及 `vision_ratio` 给 `generation` 的份额）抽样。
- **输入** 归一到 `[-1, 1]`，**target** 保持 `[0, 1]`；离散任务（semantic / instance / referring / structure）用 `NEAREST` 重采样，不产生中间色。
- `aspect_preserving_crop=True` 时输入与 target 用同一个裁剪窗口。
- 训练期增强（val 全部关闭）：
  - depth：`depth_colormaps` 多色图随机重编码并改写 prompt（需 `metric_depth/*.npy` 或可逆 target）；
  - semantic：`semantic_prompt_styles` 随机重写风格，`semantic_color_remap_prob` 概率把图例换色并同步改 target。图例来自行上的 `mapping`，没有才用 `openvocab_semantic.parse_prompt_mapping` 反解四种风格的 prompt（历史 bug：只认 `<R, G, B>`，natural/hex 行从未被增强）。
- 大规模 tar 数据：`tar_index.json` 在 `data_dir` 或 manifest 旁边时自动启用随机读取（SNVC 时代遗留，公开数据管线不需要）。
- `_collate_fn` 输出 `image / target_rgb / prompt / task` 以及 referring 的 `expression` 等附加键。

## 6. 评测数据集与下载布局的对应

| 论文表 | Dataset 类（`data/datasets.py`、`data/paper_eval_datasets.py`） | 注册表行 | 布局要点 |
|:--|:--|:--|:--|
| T2 Cityscapes | `CityscapesSegDataset` | `cityscapes`（manual） | `setup_cityscapes_official.py` |
| T3 SA-Co Gold | `SACoInstanceDataset` | `saco-gold`（HF gated） | `queries.jsonl` |
| T4 RefCOCOg | `RefCOCOgDataset(protocol="umd")` | `refcocog` | 重建的 REFER 文件 + train2014 硬链 |
| T5 ReasonSeg | `ReasonSegDataset` | `reasonseg` | `val/*.jpg + *.json` |
| T6 深度 | `NYUDepthDataset`、`IBimsDepthDataset`、`ETH3DDataset`、`DIODEDataset`、`KITTIDataset`、`nuScenesDataset` | `nyu-depth-v2-*`、`ibims1`、`eth3d-dslr-*`、`diode-val`、`kitti-depth`、`nuscenes` | NYU 经 `nyu_mat`；DIODE/ETH3D 直链 |
| T7 法向 | `NYUNormalDataset`、`DIODENormalDataset`、`ScanNetNormalDataset`、`VirtualKITTINormalDataset` | `nyu-normals`、`diode-val-normals`、`scannet-normals`、`vkitti2` | VKITTI2 不发布法向：由深度（uint16 **厘米**）+ `intrinsic.txt` 现场推导到论文坐标系 |
| App. D | `GenAIBenchDataset`、`ImgEditDataset` | `genai-bench`、`imgedit-bench` | 仅 prompt / Benchmark.tar；ImgEdit **train** 刻意不注册 |

## 7. 常见错误与定位

| 现象 | 原因 | 查哪里 |
|:--|:--|:--|
| 合并后行数骤减 | 旧版 merge 按 CWD 解析路径 | `pairs.audit.json` 的 `per_task_dropped` |
| `Validation must cover every requested training task` | 某任务没有 val 行 | `merge --val-percent` |
| 深度行被审计剔除 | 相对深度 / 灰度 / 非米制 | `rejected_non_metric_depth_rows`；检查 prepare 脚本的 `target_kind` |
| RefCOCO 图像找不到 | `--image-root` 指向 train2017 而非 train2014 链接目录 | `coco2014_links` 是否已跑 |
| Hypersim 法向告警"背向" | 导出坐标系不同 | `--normal-flip z|yz`，`manifest.json` 的 `normal_backfacing_frames` |
| 语义增强"不生效" | 行无 `mapping` 且 prompt 非四种风格 | `parse_prompt_mapping` 返回空即不增强 |
