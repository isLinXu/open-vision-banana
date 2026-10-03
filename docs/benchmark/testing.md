# Benchmark 测试记录

执行日期：2026-10-03，路径可移植性回归：2026-10-04，Asia/Shanghai；基准 main [`b490e2d3c91d5909de1ef9c9416768ec364ab884`](https://github.com/isLinXu/open-vision-banana/tree/b490e2d3c91d5909de1ef9c9416768ec364ab884)。完成 CPU 回归、真实 EMA 准备、Tiny GPU 三任务和六张 Flickr30k 两模型／三尺寸测试；未执行 SANA／FLUX 完整推理或正式官方数据集评测。

环境：CPU Python 3.12.14／torch 2.14.1+cpu；GPU Python 3.7.15／torch 1.12.0+cu116／CUDA 11.6／cuDNN 8.3.2；RTX3060 Laptop 6GB（6,441,926,656B），主机约16GB RAM。旧环境只做显式 legacy 组件测试。

## 自动验证

2026-10-03 Benchmark 相关测试为71 passed、1 skipped；2026-10-04整理后为 **87 passed、1 skipped**（约4.05s）。跳过为 Windows 无符号链接创建权限，不代表项目全部测试通过。复现命令：

```sh
python -m pytest tests/test_benchmark_checkpoints.py tests/test_benchmark_backends.py tests/test_benchmark_runner.py -q
```

覆盖固定 revision／hash／缓存篡改、下载中断与路径边界、EMA 缺失／冲突／形状／非有限拒绝、真实容器识别、禁止默认下载、无 torch 的 plan、错误配置／部分 GT／非空输出失败；online 即使篡改 hash 元数据也不能冒充 EMA。

可移植性回归覆盖缓存整体移动后离线准备和EMA校验、相对路径越界拒绝、旧绝对路径记录兼容、Windows／POSIX路径及错误文本脱敏。真实主候选缓存离线重导出后仍为84张量，EMA值及导出hash不变；12项CPU smoke再次通过。历史GPU结果未重新计时。

分析性 oracle 跑通 manifest→resize→predict→decode→原 GT 网格→全局指标：mIoU=1、AbsRel<1e−5、法线角度=0。另检验全局语义统计：1个正确像素＋9个错误像素得 mIoU=0.05、pixel accuracy=0.1；不是逐图平均0.5。ignore／mask／空有效集、零深度预测与两种法线 convention 均有反例。

正确 Turbo 输入被错误 cube-edge 解码时 AbsRel>0.5；新适配器要求明确色图及 metric-power lam/c。[protocols.py](../../scripts/benchmark/protocols.py)修正 main 历史 NYU crop 混用：480×640 使用 Eigen `[45:471,41:601]`，拒绝缩放 GT。GT全1m、预测顶部100行2m：旧区域 AbsRel=0，新区域55/426=0.129108、有效238,560像素。

crop 来源为 [Marigold 官方实现](https://raw.githubusercontent.com/prs-eth/Marigold/main/src/dataset/nyu_dataset.py)及 [ZoeDepth 官方实现](https://raw.githubusercontent.com/isl-org/ZoeDepth/main/zoedepth/utils/misc.py)（当日核验）。KITTI 使用精确 Garg 比例；ETH3D 仅项目80m cap方案、官方一致性未验证。未改原主评测源码。

## 真实图片 GPU 结果

六张固定 Flickr30k图片，两模型×长边64／128／256＝**36项**。Tiny随机模型42,492参数；预训练LR-ASPP为3,221,538参数、VOC21。顺序加载，不同时驻留；长边缩放后向下alignment8，不padding／crop。

batch1／FP32／CPU线程1；每图warm-up2次、预测5次，前后同步CUDA。predict包含后端检查、归一化与模型调用；decode只计最后一次。预测每组30次，配对预测＋解码每组6次，resize／指标／I/O不计入。

| 模型 | 长边 | predict P50／P95 ms | decode P50 ms | 配对 P50 ms | 含decode peak allocated MiB |
|---|---:|---:|---:|---:|---:|
| Tiny random | 64 | 3.142／5.147 | 2.104 | 5.238 | 2.163 |
| Tiny random | 128 | 2.744／3.372 | 0.930 | 3.593 | 8.104 |
| Tiny random | 256 | 3.483／3.915 | 2.948 | 6.497 | 31.870 |
| LR-ASPP VOC21 | 64 | 7.346／12.263 | 0.347 | 7.374 | 13.479 |
| LR-ASPP VOC21 | 128 | 9.600／11.223 | 0.465 | 10.627 | 15.311 |
| LR-ASPP VOC21 | 256 | 10.933／12.028 | 0.377 | 11.206 | 23.854 |

每图清理已释放cache，记录驻留模型及warm-up后的基线与峰值。allocated／reserved不含驱动／CUDA context，不是总显存；未记录host RAM峰值、功率、温度或端到端I/O。P95不是置信区间；同协议探索的Tiny256曾约11.3ms，时钟／热状态未受控，不构成稳定速度榜。

狗图1001773457：LR-ASPP64全背景，128产生bird／cat／sheep，256出现dog。该例展示尺寸敏感性，不能外推普遍规律；没有像素GT，不算mIoU。不同模型颜色不保证同类，随机Tiny外观不代表训练分割，LR-ASPP效果不属于OVB。

另完成Tiny合成两图×32／64／128／256×三任务＝**24项GPU**，以及两尺寸三任务＝**12项CPU**；均跑通有限预测、项目解码与产物保存。早期合成GPU运行未含独立decode计时，保留各自代码hash，不与新计时混用。[小型结果证据](evidence/laptop-test-summary-20261003.json)随仓库保存输入／代码hash与统计，不依赖原机器路径。

## 权重与剩余边界

主候选fix_best3_s1 final EMA已真实准备：84张量逐项等于来源EMA，42对LoRA、全部有限，online与EMA均不同；禁止下载后的缓存重复准备hash不变。[实物证据](evidence/selected-checkpoint-preflight-20261003.json)及[候选依据](checkpoint-selection.md)保留固定来源、导出hash和未验证项。

SANA模板plan展开成功；执行按预期exit1拒绝未核验契约，没有加载基座。权重来源／LoRA内部结构检查不证明基座revision、完整架构、加载缩放、推理协议或任务质量，inference_contract_verified=false仍有效。

复现按 [README](README.md) 使用原profile、CLI数据／权重路径与全新空输出目录；results.json的逻辑路径别名不用于重新定位输入。产物在各运行目录artifacts；prepared缓存是真正相对其JSON根目录的路径，可整体移动。

下一步先补服务器源码／patch、训练基座revision、prompts和编码，再现代CUDA单图batch1／256加载与卸载，测host RAM／总显存，最后512／20步和独立真值manifest。6GB的完整SANA sequential offload、质量／成本、配对统计及划分泄漏尚未验证。
