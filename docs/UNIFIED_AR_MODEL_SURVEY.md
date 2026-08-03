# 统一 AR 路线候选模型调研：连续 token + 共享生成通路 + 可消融

> 调研日期：2026-08-02 ｜ 背景：Janus 路线已被证伪（见 `JANUS_INFEASIBILITY_ANALYSIS.md`），若坚持统一 AR 架构验证 Vision Banana 命题，需同时满足三条件：
> **① 连续图像 token（非 VQ）｜② 输入输出共享生成通路｜③ 可与 understanding-only 预训练做消融**
> 约束：M3 Max 128GB MPS，LoRA 微调级别算力（全量预训练消融不可行，只能借助模型方发布的中间 checkpoint / 论文消融数据）。

## TL;DR

- 🟢 **首选实验平台：商汤 SenseNova U1（NEO-Unify 架构，8B-MoT）**——2026 年开源，三条件全满足：encoder-free（无 VE 无 VAE，像素直进直出）、理解/生成共享 MoT 骨干、训练分四阶段（理解预热→生成预训练→联合→SFT）天然提供消融臂，且技术报告明确引用 Vision Banana 的"生成即理解"命题。8B bf16 约 16GB，MPS 可 LoRA。
- 🟢 **次选：Orthus-7B（SJTU，已开权重 base+instruct）**——连续 patch 特征 + 模态特定头（diffusion head 出图），"lossless visual signal for understanding and generation"即共享通路；7B MPS 可训。
- 🟡 **消融臂专用：MAR（纯生成预训练，0.4–0.8B）**——连续 token 生成-only 模型，是"generation-only pretraining"臂的最干净样本，MPS 上连全量微调都可行。
- 🟡 **OneCAT-3B（encoder-free，已开源）**——架构理念与 NEO 类似且更轻，token 性质需以技术报告核实后入选。
- 🔴 **UniFluid / BAGEL / Skywork UniPic**——连续 token ✓ 但理解侧走 SigLIP 独立编码器，条件 ② 只满足一半；UniFluid 未见官方权重发布。可作对照组与消融证据引用，不宜作主平台。
- **现成对照臂**：你已有的 Janus 实验 = "理解侧预训练污染"臂；SSD-1B/FLUX 扩散实验 = "生成预训练（扩散机制）"臂。统一 AR 平台只需补上"连续 token 生成预训练"这一格。

---

## 一、候选模型总览（三条件合规矩阵）

| 模型 | 发布方/年份 | 规模 | ①连续token | ②共享通路 | ③可消融 | 开源状态 | MPS可行性 |
|---|---|---|---|---|---|---|---|
| **SenseNova U1** 🟢 | 商汤 2026 | 8B-MoT / A3B(38B总) / NEO 2B | ✅ 像素 patch（32×32px conv 直映，无 VQ） | ✅ **encoder-free**，理解生成同一表示空间（重建 PSNR 31.56 ≈ FLUX VAE 32.65） | ✅ 四阶段训练（理解预热/生成预训练/联合/SFT）+ 技术报告含消融 | ✅ 权重+代码+技术报告全开 | 🟢 8B bf16≈16GB，LoRA 可行；推理栈为 CUDA 优化需移植 |
| **Orthus** 🟢 | SJTU 2024 | 7B | ✅ 连续 patch 特征，免 VQ 信息损失 | ✅ 同一视觉特征供理解与生成，diffusion head 出图 | 🟡 有 base（预训练）与 instruct 两版权重，可分臂 | ✅ base+instruct 权重开源，商用许可 | 🟢 7B 可 LoRA；依赖魔改 transformers+flash-attn，MPS 需适配 |
| **MAR** 🟡 | He et al. 2024 | 0.4–0.8B | ✅ 连续 + diffusion loss（无 VQ 开山之作） | ✅ 纯生成通路（无理解编码器） | ✅ **本身就是 generation-only 臂** | ✅ 权重+代码开源 | 🟢 极小，MPS 全量微调都行 |
| **OneCAT** 🟡 | OneCAT-AI 2025 | 3B | 🟡 Patch Embedding 直映（无 ViT/VAE），token 连续/离散性质待核实 | ✅ decoder-only 统一 AR，推理不挂外部 diffusion | 🟡 待核实 | ✅ 权重+代码开源 | 🟢 3B 很轻 |
| **UniFluid** 🟡 | Google 2025 | 基于 Gemma（2B/9B 量级） | ✅ 连续 VAE token + per-token diffusion head | ❌ 理解走 SigLIP、生成走 VAE，双编码器 | 🟡 论文含 loss 平衡/任务互益消融（可直接引用为证据） | 🔴 未见官方权重发布（需 GitHub 复核） | 权重不可得则不可行 |
| **Skywork UniPic** 🟡 | 昆仑万维 2025 | 1.5B | ✅ 生成侧 MAR 连续 | ❌ 理解 SigLIP2 / 生成 MAR 解耦（Harmon 改） | 🟡 全流程代码+报告开源 | ✅ 权重+代码+报告全开 | 🟢 1.5B 极轻 |
| **BAGEL** 🟡 | 字节 2025 | 7B+7B MoT | ✅ 生成侧 VAE 连续 latent | ❌ 理解 SigLIP / 生成 VAE 双编码器 | 🟡 训练代码开源 | ✅ 权重+训练代码开源 | 🟡 14B 总参，MPS LoRA 勉强 |
| **Emo2/Emu2** 🔴 | BAAI 2023 | 37B | ✅ next-embedding 连续回归 | 🟡 生成经冻结扩散解码器 | ❌ | 🟡 权重开、训练代码不全 | 🔴 37B 超出 MPS |
| **Transfusion** 🔴 | Meta 2024 | 7B | ✅ | ✅ 单 transformer | ❌ | 🔴 无官方权重 | 不可行 |
| **Ming-Lite-Uni** 🔴 | inclusionAI 2025 | ~14B | ✅ 连续 | ❌ 冻结 MLLM + 可训扩散，通路分离 | ❌ | 🟡 alpha | 🟡 |
| **Janus / Emu3 / Chameleon / Show-o / TokenFlow / VILA-U** 🔴 | — | — | ❌ 离散 VQ | 各异 | — | — | 已被你证伪/不满足① |

## 二、推荐实验矩阵

把"什么预训练赋予理解能力"分解为 2×2 消融，你手里已有的实验可以直接填入三格，统一 AR 只需补最后一格：

```
                        理解侧预训练
                   无                有
生成预  无    │ （空白基线）      │ ✅ 已有：Janus 理解路径
训练    连续AR│ 🟡 待补：本次目标  │ 🟡 对照：UniFluid/BAGEL
        扩散  │ ✅ 已有：FLUX/SSD-1B│ （NBP 论文本身）
```

**具体分工：**

1. **主平台：SenseNova U1-8B-MoT**（若 2B NEO 版放出则更优先——MPS 上迭代快 4 倍）
   - 用其"生成预训练"阶段 checkpoint（若发布）直接做 Vision Banana 式 RGB 指令微调 → 测 depth/normal/seg 的 held-out；
   - 与"理解预热"checkpoint 同配方微调 → 构成 ③ 的消融对；
   - 像素直出（MLP 预测 32×32 patch）意味着 colormap 不经过任何 VQ/VAE 瓶颈——P0-3 信号摧毁问题在结构上不存在。
2. **独立验证：Orthus-7B**——不同团队、不同 tokenizer 路线的同构架构，用于排除"U1 特有实现细节"的混杂。
3. **下界探针：MAR（0.4–0.8B）**——纯生成预训练、无语言/理解监督，指令微调后若能超过随机基线，即为"生成预训练含可迁移理解"的最小证据；MPS 上一周可出结果。
4. **对照引用（无需训练）**：UniFluid 论文的 loss-balance 消融（生成↔理解互益区间）与 BAGEL 报告，作为联合训练配方参考。

## 三、风险与核实清单

- 🔴 **U1 的中间阶段 checkpoint 是否随权重发布**——消融设计的关键依赖，需到 HF `sensenova/sensenova-u1` collection 逐一核对；若只发布最终版，③ 降级为"引用其技术报告消融 + 自训 MAR/Orthus 分臂"。
- 🟡 **MPS 移植成本**：U1 推理栈（LightLLM/LightX2V、FlashAttention3）CUDA 导向，训练代码预计基于 PyTorch 标准算子可移植，但需预留 1–2 天适配；Orthus 依赖魔改版 transformers，是已知的移植痛点（Janus 当时踩过同类坑）。
- 🟡 **OneCAT token 性质**与 **UniFluid 权重状态**——两处标记"待核实"，入选前先到 GitHub 确认。
- 🟢 **License**：U1 / Orthus / UniPic 均声明可商用级开源，MAR 为研究许可，均无阻碍。

## 四、来源

- SenseNova U1：技术报告 arXiv:2605.12500，HF collection `sensenova/sensenova-u1`，GitHub `OpenSenseNova/SenseNova-U1`
- Orthus：arXiv:2412.00127，GitHub `SJTU-DENG-Lab/Orthus`（HF 有 base/instruct 权重）
- UniFluid：arXiv:2503.13436（Google DeepMind，Gemma 骨干，连续 token + SigLIP 理解前缀）
- Skywork UniPic-1.5B：HF `Skywork/Skywork-UniPic-1.5B`，GitHub `SkyworkAI/UniPic`
- OneCAT-3B：HF `onecat-ai/OneCAT-3B`，GitHub `onecat-ai/onecat`
- MAR：arXiv:2406.11838（"Autoregressive Image Generation without Vector Quantization"）
- 背景归因：`docs/JANUS_INFEASIBILITY_ANALYSIS.md`
