# 模型仓库与全量 checkpoint 发现

## 当前待测范围

A/B仓库`ckpt/`目录下**全部完整、可加载checkpoint**，包括普通、`_best`、final等实际发布类型；不设步数上限，不再只列此前的22500/25000/27500。共同起点作为对照。后续新增目录、同路径替换权重/配置或运行元数据，均需固定新revision重新核验。

下表是当前查询快照，不代表已完成评分；后续以重新扫描结果为准。

| 模型 | 模型仓库 | 固定快照/ckpt目录 | 本次发现 |
|---|---|---|---|
| I108-A · lr3e-5 | [khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3) | [`624198cc6896`](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt) | 22份，最高42,500步 |
| I108-B · lr1e-5 | [khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3) | [`c974c8bcdfa3`](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt) | 22份，最高37,500步 |
| 共同起点 Fusion-C1-KQ0r128-60-40 | [Melo111/Fusion-C1-KQ0r128-60-40](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40) | [`9d5b9b8ede04`](https://huggingface.co/Melo111/Fusion-C1-KQ0r128-60-40/tree/9d5b9b8ede04d7d9f4a4ff8a6a75a4b734ab1b4d) | 完整merged pipeline，新增训练0步 |

## 跟踪新增和最新更新

```bash
# 默认扫描A/B所有ckpt。仅读取HF文件元数据，不下载大权重。
python scripts/discover_external_eval_checkpoints.py --output /absolute/checkpoints-latest.json
# 后续重复同一命令，会与上一次输出比较；也可前台持续轮询：
python scripts/discover_external_eval_checkpoints.py --output /absolute/checkpoints-latest.json --watch-seconds 600
```

每次先解析HEAD为固定commit，再按该commit列举目录。`content_id`由仓库、权重LFS SHA256、配置Git blob和checkpoint运行元数据blob构成；仅README更新不产生新内容，同路径换权重/配置/元数据会被发现。相同内容的路径别名仍保留在全量清单，内容发现按identity去重。缺失文件、无权威hash或其他权重布局列入`unresolved_checkpoints`，不默认为已完成。

`new_or_updated_content_ids`表示新增内容发现；`seen_content`不等于测过。所有条目的`evaluation_status`均要求核对验收记录，复用时还需匹配共同起点、数据、画布和评分协议。扫描器不自动跳过未测权重，也不自动下载或启动GPU队列。持续轮询是用户显式运行的前台命令，本次没有启动常驻监控。

机器可读参考快照：[i108-checkpoints.snapshot.json](../configs/evaluation/i108-checkpoints.snapshot.json)。只选择实际可加载的模型权重；优化器状态不是独立模型。普通与best同目录同时有bin/safetensors时记录加载器优先的safetensors序列化。

## 当前快照的全部 checkpoint 链接

发现数量和发布状态不等于新双标注协议的验收状态；已完成旧协议的条目仍需按新指标需求核对。

| 模型 | 步数 | 类型 | 固定revision目录 |
|---|---:|---|---|
| A | 2500 | _best | [ckpt/checkpoint-2500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-2500_best) |
| A | 5000 | 普通/其他 | [ckpt/checkpoint-5000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-5000) |
| A | 5000 | _best | [ckpt/checkpoint-5000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-5000_best) |
| A | 7500 | _best | [ckpt/checkpoint-7500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-7500_best) |
| A | 10000 | 普通/其他 | [ckpt/checkpoint-10000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-10000) |
| A | 10000 | _best | [ckpt/checkpoint-10000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-10000_best) |
| A | 12500 | _best | [ckpt/checkpoint-12500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-12500_best) |
| A | 15000 | 普通/其他 | [ckpt/checkpoint-15000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-15000) |
| A | 15000 | _best | [ckpt/checkpoint-15000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-15000_best) |
| A | 17500 | _best | [ckpt/checkpoint-17500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-17500_best) |
| A | 20000 | 普通/其他 | [ckpt/checkpoint-20000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-20000) |
| A | 22500 | _best | [ckpt/checkpoint-22500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-22500_best) |
| A | 25000 | 普通/其他 | [ckpt/checkpoint-25000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-25000) |
| A | 30000 | 普通/其他 | [ckpt/checkpoint-30000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-30000) |
| A | 30000 | _best | [ckpt/checkpoint-30000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-30000_best) |
| A | 32500 | _best | [ckpt/checkpoint-32500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-32500_best) |
| A | 35000 | 普通/其他 | [ckpt/checkpoint-35000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-35000) |
| A | 35000 | _best | [ckpt/checkpoint-35000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-35000_best) |
| A | 37500 | _best | [ckpt/checkpoint-37500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-37500_best) |
| A | 40000 | 普通/其他 | [ckpt/checkpoint-40000](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-40000) |
| A | 40000 | _best | [ckpt/checkpoint-40000_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-40000_best) |
| A | 42500 | _best | [ckpt/checkpoint-42500_best](https://huggingface.co/khazic/ovb-i108-a-lr3e-5-r64-2x8-s42_20261109r3/tree/624198cc6896982c098d7a5654d28cdc19c6e0ce/ckpt/checkpoint-42500_best) |
| B | 2500 | _best | [ckpt/checkpoint-2500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-2500_best) |
| B | 5000 | 普通/其他 | [ckpt/checkpoint-5000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-5000) |
| B | 5000 | _best | [ckpt/checkpoint-5000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-5000_best) |
| B | 7500 | _best | [ckpt/checkpoint-7500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-7500_best) |
| B | 10000 | 普通/其他 | [ckpt/checkpoint-10000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-10000) |
| B | 10000 | _best | [ckpt/checkpoint-10000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-10000_best) |
| B | 12500 | _best | [ckpt/checkpoint-12500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-12500_best) |
| B | 15000 | 普通/其他 | [ckpt/checkpoint-15000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-15000) |
| B | 15000 | _best | [ckpt/checkpoint-15000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-15000_best) |
| B | 17500 | _best | [ckpt/checkpoint-17500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-17500_best) |
| B | 20000 | 普通/其他 | [ckpt/checkpoint-20000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-20000) |
| B | 20000 | _best | [ckpt/checkpoint-20000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-20000_best) |
| B | 22500 | _best | [ckpt/checkpoint-22500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-22500_best) |
| B | 25000 | 普通/其他 | [ckpt/checkpoint-25000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-25000) |
| B | 25000 | _best | [ckpt/checkpoint-25000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-25000_best) |
| B | 27500 | _best | [ckpt/checkpoint-27500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-27500_best) |
| B | 30000 | 普通/其他 | [ckpt/checkpoint-30000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-30000) |
| B | 30000 | _best | [ckpt/checkpoint-30000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-30000_best) |
| B | 32500 | _best | [ckpt/checkpoint-32500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-32500_best) |
| B | 35000 | 普通/其他 | [ckpt/checkpoint-35000](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-35000) |
| B | 35000 | _best | [ckpt/checkpoint-35000_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-35000_best) |
| B | 37500 | _best | [ckpt/checkpoint-37500_best](https://huggingface.co/khazic/ovb-i108-b-lr1e-5-r64-2x8-s42_20261109r3/tree/c974c8bcdfa3609f7af75100e019bc4c42c4a922/ckpt/checkpoint-37500_best) |

## 历史对照仓库

以下来自既有服务器结果的谱系记录，用于回归/历史比较；不取代A/B全量范围。`_retest`是评测标识，不是另一个HF仓库。各权重实际步数、revision、EMA/online及merged起点仍应按原始记录核验。

| 模型 | 仓库 | 已有记录步数 |
|---|---|---:|
| BL_retest | [khazic/ovb-ft_bl_t1_cube_20261031_2014](https://huggingface.co/khazic/ovb-ft_bl_t1_cube_20261031_2014/tree/9b1f65949cd2360d10613a6304cc127a2b96efcb) | 10000 |
| C1_retest | [khazic/ovb-i88_c1_depth_cube100_20261030_1602](https://huggingface.co/khazic/ovb-i88_c1_depth_cube100_20261030_1602/tree/be6d643b8b8f58aa641c318a4ac02a1ff95b69cf) | 10000 |
| E系列起点来源（父adapter） | [khazic/ovb-e2b_cap10k_s42_20261026_1016](https://huggingface.co/khazic/ovb-e2b_cap10k_s42_20261026_1016/tree/036bc7285371e24b9e52ab1b0ad9aea9ea337f27) | 0 |
| M0 | [khazic/ovb-i97-m0-s42_20261102_2007](https://huggingface.co/khazic/ovb-i97-m0-s42_20261102_2007/tree/99916779c41958fa1dbdc38d11a21ad6bb6aa646) | 20000 |
| M1 | [khazic/ovb-i97-m1-s42_20261102_2007](https://huggingface.co/khazic/ovb-i97-m1-s42_20261102_2007/tree/823fbda4fd200aee892266ebf13db9ebad824ca4) | 20000 |
| Q0 | [khazic/ovb-i97-q0-s42_20261102_2007](https://huggingface.co/khazic/ovb-i97-q0-s42_20261102_2007/tree/27d1dd02ba008a85fa8107ddbb4b0dc7827bc1e3) | 20000 |
| Q1 | [khazic/ovb-i97-q1-s42_20261102_2007](https://huggingface.co/khazic/ovb-i97-q1-s42_20261102_2007/tree/3ae835bafda900711b9171d883a7f2120dda1b6f) | 20000 |
| K-Q0-r64 | [khazic/ovb-i97-k-q0-r64-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-k-q0-r64-s42_20261103_queue/tree/4f5936dfe6ee2bdc08d091c79d077eeb0cd54ab9) | 20000 |
| K-Q1-r64 | [khazic/ovb-i97-k-q1-r64-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-k-q1-r64-s42_20261103_queue/tree/b10fe9ea00b6a2563d365b683966e82ab7967e39) | 20000 |
| K-Q0-r128 | [khazic/ovb-i97-k-q0-r128-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-k-q0-r128-s42_20261103_queue/tree/1b84ff35d9989f3126c6acc86010df710d89585c) | 20000 |
| F-M1-lr1e-6 | [khazic/ovb-i97-f-m1-lr1e-6-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-f-m1-lr1e-6-s42_20261103_queue/tree/8fa0f888ed9e4413d8c615603a025043383e7c5b) | 15000 |
| F-M1-lr2e-6 | [khazic/ovb-i97-f-m1-lr2e-6-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-f-m1-lr2e-6-s42_20261103_queue/tree/97422c3ba9df64293224948a415044d7d69e3351) | 15000 |
| F-M1-lr5e-6 | [khazic/ovb-i97-f-m1-lr5e-6-s42_20261102_2007](https://huggingface.co/khazic/ovb-i97-f-m1-lr5e-6-s42_20261102_2007/tree/c915f2372e163d6bb133ea4f1a7d875b61342180) | 15000 |
| F-M1-lr1e-5 | [khazic/ovb-i97-f-m1-lr1e-5-s42_20261103_queue](https://huggingface.co/khazic/ovb-i97-f-m1-lr1e-5-s42_20261103_queue/tree/eb5b81496b693600cf92554bd5204378e53cf4ff) | 15000 |

E系列父adapter的链接不能当作独立merged pipeline直接替代起点；按既有谱系还原。其他本地融合条目按融合配方和权重SHA追溯，已发布的60:40完整基座见主表。
