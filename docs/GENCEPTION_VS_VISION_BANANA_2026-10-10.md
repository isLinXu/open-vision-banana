# GenCeption 官方代码对照 Vision Banana（2026-10-10）

对照对象：

- DeepMind 发布代码：[google-deepmind/representations4d/representations4d/genception](https://github.com/google-deepmind/representations4d/tree/main/representations4d/genception)
- 论文：[Video Generation Models are General-Purpose Vision Learners](https://arxiv.org/abs/2607.09024)（ECCV 2026）
- 本仓库：open-vision-banana，复现的是 Vision Banana（[2604.20329v3](https://arxiv.org/abs/2604.20329)）的**图像生成器 → 视觉理解**方法

**原则：主线是复现 Vision Banana。GenCeption 只提供「表示写进数据、少改主干、训练稳健」的对照，不换基座、不搬 JAX/WAN、不把视频前馈当成本仓库的评测协议。**  
仓库里已有的 `docs/Open_GenCeption_PRD_v1.0.md` 是按论文推的产品稿；下面以**已发布推理代码**为准，和那份 PRD 不一致的地方以本文为准。

---

## 1. 先把对象说清楚

口头上说「Genception 是视频编辑、Vision Banana 是图像编辑」方向对，细节要收紧：

| | Vision Banana / 本仓库 | GenCeption 论文 + 发布代码 |
|--|------------------------|----------------------------|
| 基座 | 图像生成 / 图像编辑（SANA、FLUX.1-dev、Kontext 等） | **文本生成视频** WAN 2.1（1.3B / 14B） |
| 条件 | 源图 + 指令，走 img2img / in-context tokens | **干净输入视频潜变量** + 预计算文本嵌入 |
| 发布物 | 训练 + 多步推理 + 官方划分评测 | **只有 JAX 推理**。无训练循环、无数据、无评测脚本 |
| 图像条件 | Kontext / Qwen-Edit 等是真编辑通路 | `encoder_hidden_states_image` 直接 `NotImplementedError('Image-to-video is not yet implemented.')` |
| 单图怎么跑 | 原生一张图 | `run_inference.py` 把一张图 **在时间维复制成 T 帧**（论文默认 81，demo 用 5） |

所以 GenCeption 不是 Runway 那种指令视频编辑器。它是：把 T2V 扩散改成「视频进、感知 RGB 出」的前馈映射。单图 demo 只是把静帧铺成假视频，好塞进 3D RoPE。本仓库已经有真图像条件，**不要再学「复制 81 帧」来冒充视频先验。**

论文 Related Work 写得很直白：Vision Banana 在 2D 图像空间；他们扩到原生视频，并且用前馈替换多步生成。深度编码还写了 `similar to [18]`（就是 Vision Banana）。关系是**同族后作**，不是另一套无关方法。

---

## 2. 发布代码实际长什么样

目录只有推理栈，Apache-2.0：

| 文件 | 作用 |
|------|------|
| `pipeline.py` | `GenCeptionPipeline`：VAE encode → 单步 DiT → VAE decode |
| `model.py` | WAN DiT + 两路可学习 token（2D / 3D 关键点） |
| `vae.py` | WAN 视频 VAE，时间 4×、空间 8× |
| `attention.py` / `embeddings.py` / `normalization.py` | 3D RoPE、交叉注意力、RMSNorm |
| `utils.py` | 权重、视频 IO、**cube-edge 深度编解码** |
| `run_inference.py` | 单图铺成短视频的冒烟脚本 |

没有：optimizer、dataloader、loss、LoRA、评测 adapter。权重在 GCS：`genception_1.3b_transformer.npz`、`genception_14b_transformer.npz`、单独 VAE。文本侧要求离线 UMT5-XXL 嵌入（`[B, seq, 4096]`），运行时不加载 22GB 文本塔。

### 2.1 前馈怎么做（这是核心，不是「少采几步」）

`pipeline.__call__` 和论文 3.2 一致：

1. 输入视频（或静帧复制）进冻结 VAE，取 posterior **mode**，不加噪声。
2. 时间步固定 **`t = 0`**（`diffusion_single_step: 0`），表示 Rectified Flow 已经走到干净端。
3. DiT **只走一次**。`num_inference_steps` 参数还在，但不用来做 ODE。
4. `WanModel` 最后一行是 `return -video_latents, token1_pred, token2_pred`。  
   预训练预测 `v = ε − x0`，取负得到 `−v = x0 − ε`，和目标潜变量同号。
5. 直接 VAE decode，没有 scheduler.step。

这和本仓库现在的评测相反：SANA 20 步、FLUX 20 步、Kontext 28 步，都是在噪声桥上积分。GenCeption 把「生成采样」收成「特征提取 + 一次映射」。

### 2.2 稠密任务：RGB 空间 + 潜空间 L2

和 Vision Banana 一样，深度 / 法向 / 分割都落在 3 通道 RGB。不同处：

- **监督在 VAE 潜空间做 L2**，不是像素 MSE，也不是 AbsRel / 角误差。
- 任务差别尽量写进**数据表示**，不写进 loss。深度：先按视频中位数消尺度，再 `d' = clip(α log(d+1), 0, 1)`，再走 **Banana coloring**（就是 cube-edge：黑→红→黄→绿→青→蓝→品红→白）。
- 发布代码 `decode_depth(..., depth_scaling_factor=0.4)` 反的是这套 **相对深度 + log + cube-edge**，不是 Vision Banana 的度量幂变换 `λ=-3, c=10/3`。

高维相机位姿压成 3 通道「Rothko raymap」：画面中心放射线原点，外围放方向。这是数据格式技巧，不是新 decoder。

### 2.3 稀疏任务：往序列上接 token（他们自己说这条有代价）

`model.py` 在视频 token 后面拼两路可学习 query：

- token1：每帧 1 个，输出 1 维（配置里像 2D 关键点）
- token2：每帧 1 个，输出 3 维（3D 关键点）
- 时间 RoPE 用位置插值压进预训练 `T'`；空间 RoPE **可学习**（空间位置未知）
- MLP head 在 DiT 之后

论文 ablation 写得很重：联合训练会严重伤 3D 关键点，也拖累部分稠密任务。原因是坐标回归离开了像素预训练域，而且从零学的 token 打乱预训练注意力。结论是：**对预训练主干改得越少越好。**

### 2.4 训练配方（只在论文里，代码没有）

- 分辨率 480×832，81 帧，24 fps
- 全量微调 WAN（不是 LoRA）
- Adam `5e-5`，15k step，250 warmup，batch 64，256 块 TPU v6e
- **梯度裁剪 + 梯度丢弃**（范数过大的 batch 整份扔掉）
- 合成数据为主：7500 段 RenderPeople + CMU mocap；深度再混 TartanAir / VKITTI / MVS Synth
- 指代分割例外，用了 MeViS / RefCOCO / YouTube-VOS 真数据
- 离线缓存：RGB 潜变量、目标潜变量、文本嵌入

推理成本（论文）：1.3B 约 13.6 FPS / 15.3 GB；14B 约 8.0 FPS / 42.8 GB（81 帧）。文本塔 10 GB 可卸到内存。

---

## 3. 和本仓库的对齐，以及不该对齐的地方

### 已经同族、不必再抄的

1. **稠密输出进 RGB。** 这边的 `DepthRGBMapper` / `NormalRGBMapper` / 分割最近色已经是这条路。
2. **Cube-edge。** GenCeption 注释里就叫 Banana coloring。继续用论文 Eq.1 + 八顶点七段即可，不要换成他们的 log-relative，除非实验题目改成相对深度。
3. **文本切任务。** 两边都是 prompt 选模态，不是换 head。
4. **合成几何数据。** Vision Banana 也强调深度全合成、评估集不进训练。

### 看起来像、其实不能搬的

| GenCeption 做法 | 若直接搬到本仓库会怎样 |
|-----------------|------------------------|
| 静帧复制成 81 帧 | 假时间维。浪费算力，也测不出视频先验。Kontext / img2img 已经是真图像条件 |
| 全量微调 14B WAN | 和当前 24GB + LoRA 路线冲突；也不是 Vision Banana 论文披露的更新范围 |
| 可学习关键点 token | 他们自己证明会伤联合训练。图像 LoRA 更不该先加这条 |
| JAX + 预计算 UMT5 | 另一套运行时。现有 PyTorch / diffusers 栈没有理由换 |
| ETH3D AbsRel 0.037 当目标 | 他们是相对深度、中位数归一、视频模型、可能不同 crop。不能和 NYU Eigen+10m / ETH3D 80m 的 δ1 对打 |
| 单步 t=0 当前就替换 20 步评测 | 当前 SANA/FLUX 多步闸门还没过。先换推理公式，分不清是表示失败还是采样失败 |

### 值得借的是「问题」，不是文件

1. **表示写进数据，loss 保持一条。**  
   他们用中位数消尺度 + log，是为了继续用潜空间 L2，不用 scale-invariant depth loss。这边度量 cube-edge 在远处接近饱和，AbsRel 会被少数像素炸掉（DIODE/iBims 到 10³）。可做的对照是：同一套 LoRA，只改 GT 编码（度量幂 vs 中位数+log），loss 不动。这是借鉴，不是换基座。

2. **前馈作为晋级后的效率实验，不是现在的主协议。**  
   论文对齐评测应继续多步。闸门过了再试：干净源图潜变量、固定小 t、一次前向、输出按 flow 符号取负。本仓库 `conditioning.py` 里已有 `single_step_x0` 雏形，和他们 `t=0, return -v` 是亲戚，不要再 fork 一份 WAN。

3. **少改主干。**  
   他们加 token 是反面教材。继续 LoRA / 原条件通路，比自造 query token 更接近「最小改动」。

4. **离线缓存潜变量和文本。**  
   合成深度已经有 prepare 脚本。把 VAE latent 和 T5/Gemma 嵌入缓存下来，能把 4090 上的有效步数打上去，而不用换 JAX。

5. **稳定训练：clip + 丢异常 batch。**  
   论文说这两项对质量是必要的。便宜，且不改变方法声称。

6. **specialist vs generalist 分开报。**  
   他们 Table 1 两列都给。这边现在是各任务各 LoRA。共享权重之前，不要把拼接 adapter 写成 unified model。

7. **层迁移消融。**  
   他们 Figure 9：随机初始化几乎不学，预训练层越多越好。这边若要证明「生成预训练有用」，应对同一数据比较：随机 DiT / 冻结特征 probe / LoRA / 全量。现在还没有这个因果实验。

---

## 4. 数字为什么不能横着比

GenCeption-L specialist（合成数据，论文 Table 1/2）：Sintel AbsRel 0.130、KITTI 0.048、ETH3D 0.037。  
本仓库最好的已测 4-set δ1 大约 0.16，ETH3D AbsRel 0.54（SANA metric）。

差看起来很大，但不是「再抄一遍前馈就能齐」：

- 他们评的是**相对、中位数归一**的深度；这边标题指标是**米制 + Eigen/Garg/80m**。
- 他们是 14B 全量微调的视频 DiT；这边是 1.6B/12B 的图像 LoRA，20–28 步采样。
- 评估集和 crop 不完全相同（他们还有 Sintel / Goliath 视频协议）。
- 发布代码没有评测脚本，无法在同一 JSON 协议下复算。

要借鉴，就在**同一套 `evaluate_paper_benchmarks.py` 协议**上做消融，不要把论文表抄进本仓库主结果。

---

## 5. 已按 Vision Banana 主线落地的借鉴（不是 Open-GenCeption）

保持主线：图像编辑 / 图像生成基座、RGB 任务、论文划分评测、不编造 Table 1。默认协议没有改。

| 借鉴 | 落地位置 | 默认 |
|------|----------|------|
| 表示写进数据，loss 仍是一条 flow L2 | `DepthRGBMapper(curve=...)`：`metric_power`（论文 Eq.1）或 `log_relative`（中位数 + `clip(α ln(d+1))` + 同一 colormap） | `metric_power` |
| 训练 / 评测同一套曲线 | `TrainingConfig.depth_curve`、`ManifestVisionDataset` 按曲线重编码、`--depth-curve` / `--depth-log-alpha` | 论文曲线 |
| clip + 丢掉爆炸梯度 | `TrainingConfig.drop_grad_norm` + `drop_oversized_gradients`。0 关闭 | 关闭（`0.0`）。启用时必须 `> max_grad_norm`，建议 `10.0` |
| 少改主干 | 没有加关键点 token，也没有接 WAN | LoRA / 原条件通路 |
| 离线 latent 缓存 | 已有 `LatentCacheDataset` | 按需打开 |
| 单步前馈 | 仍是闸门之后的旁路（`conditioning.single_step_x0`） | 评测继续多步 |

**怎么做编码对照（仍是 Vision Banana 数字协议）：**

```text
# 训练（可选对照，不替换当前 metric cube-edge 实验）
# data.depth_curve: log_relative
# training.drop_grad_norm: 10.0

# 评测：曲线必须和训练一致。log_relative 的未对齐 AbsRel 不能当论文指标。
python scripts/evaluate_paper_benchmarks.py --benchmark depth \
  --depth-curve metric_power   # 论文协议
```

同一 checkpoint 换 `--depth-curve` 只是解码对照，不是新方法。要看「是不是表示问题」，需要用同一基座、同一 loss、只改 `depth_curve` 再训一档，再在 `evaluate_paper_benchmarks.py` 的 NYU Eigen+10m 协议上比 δ1 / AbsRel。

还没做、也不该现在做：把 WAN / 81 帧铺图 / 关键点 token / JAX 迁进 `src/open_vision_banana`；用 GenCeption 的 ETH3D 0.037 当成本仓库目标；用单步 t=0 替换当前 20/28 步论文评测。

---

## 6. 和旧 PRD 的勘误

`docs/Open_GenCeption_PRD_v1.0.md` 里这些点和发布代码不一致，以后不要当实现规格：

- 发布物是推理 + 权重，不是可训练框架。
- I2V 未实现；单图是时间复制。
- 深度是 **log-relative + cube-edge**，默认 `c=0.4`，不是「三通道复制 log」这么简单。
- 输出取负发生在 **DiT 末层**，pipeline 不再做 ODE。
- 稀疏 token 始终挂在 1.3B/14B 配置上；论文还说联合训练会伤它。
- 文本是离线 4096-d 嵌入，不是运行时再跑 T5。

---

## 7. 一句话

GenCeption 证明的是：**视频生成预训练 + 尽量少改的前馈微调 + 把任务差异塞进 RGB/数据**，可以在他们的协议上打到专用模型附近。  
Vision Banana 要证明的是另一句话：**图像生成/编辑模型经少量视觉对齐后，仍是通用理解器并保住生成。**  

可以借「数据里消歧、loss 保持一条、少改主干、过闸后再试单步」。不能借 WAN 权重、假视频铺帧、稀疏 token，也不能用他们的 ETH3D 0.037 当成本仓库的目标数。
