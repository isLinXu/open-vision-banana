# Open-GenCeption PRD 产品需求文档

> **文档版本**: v1.0  
> **基于论文**: *Video Generative Models are Generalist Vision Learners* (arXiv:2607.09024)  
> **目标**: 构建一个完全开源、可复现的 GenCeption 通用视频感知模型  
> **适用项目**: 可在 open-vision-banana 框架内作为视频感知扩展模块实现，亦可独立成项目

---

## 1. 项目概述 (Executive Summary)

### 1.1 背景与动机

计算机视觉长期停留在"专用模型"阶段：每个任务（深度估计、分割、位姿估计）都需要独立的架构和训练流程。这与 NLP 领域通过下一个 token 预测实现的统一基础模型范式形成鲜明对比。

GenCeption 论文提出了一个革命性的洞察：**大规模文本到视频生成模型本身就是强大的通用视觉学习器**。视频生成过程中内化的时空先验、3D 几何理解、物理交互规律以及原生视觉-语言对齐，可直接迁移到多种下游感知任务，无需昂贵的迭代采样。

本项目目标是构建一个 **Open-GenCeption** —— 完全开源、社区可复现的通用视频感知框架，将预训练的视频扩散主干重构为高效的前馈感知模型，支持深度估计、表面法线估计、前景分割、表达引用分割、2D/3D 关键点预测、相机位姿估计等多样化任务。

### 1.2 核心创新点

| 维度 | 传统方法 | GenCeption 范式 |
|------|---------|----------------|
| 预训练目标 | 对比学习 / 掩码自编码 | 文本到视频生成 |
| 架构模式 | 任务特定编码器+解码器+损失 | 统一主干+统一头部+统一损失 |
| 推理方式 | 多步迭代扩散采样 | 单步前馈 |
| 任务扩展 | 新增模块+重新设计架构 | 新增数据格式+文本提示 |
| 数据效率 | 数百万级真实标注 | 数千合成视频即可达标 |
| 泛化能力 | 训练域内 | 零样本跨域、跨类别 |

### 1.3 项目目标

- **短期（M1-M3）**: 实现核心架构，在深度估计单任务上验证前馈化改造可行性
- **中期（M4-M6）**: 扩展至 6+ 视觉任务，在多基准上达到可接受性能
- **长期（M7+）**: 完整复现论文涌现行为（sim-to-real 迁移、分布外泛化），发布可训练权重

### 1.4 与现有工作的关系

- **Vision Banana** (2604.20329): 基于图像扩散模型（FLUX）的通用视觉学习器，本项目扩展至**视频领域**
- **Diception**: 基于图像扩散的通用视觉模型，本项目原生支持**时间一致性**
- **THFM** (2603.25892): 论文作者的前作，人类中心 4D 感知模型，本项目扩展至**通用物体类别**
- **DepthCrafter / NormalCrafter**: 专注于单一稠密任务的扩散再利用，本项目追求**统一多任务**

---

## 2. 核心架构设计 (Core Architecture)

### 2.1 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Open-GenCeption 架构总览                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   输入视频 [B, T, 3, H, W]                                               │
│      │                                                                  │
│      ▼                                                                  │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐    │
│   │  VAE Encoder│    │ Text Encoder│    │  可学习 Token (稀疏任务) │    │
│   │  (冻结)     │    │  (冻结)     │    │  [B, T, N_learn, D]     │    │
│   │  时空压缩   │    │  文本嵌入   │    │  3D RoPE + MLP 解码      │    │
│   │  8×空间     │    │  [B, L, D]  │    │                         │    │
│   │  4×时间     │    │             │    │                         │    │
│   └──────┬──────┘    └──────┬──────┘    └────────────┬────────────┘    │
│          │                  │                        │                 │
│          ▼                  ▼                        ▼                 │
│   ┌─────────────────────────────────────────────────────────┐         │
│   │              DiT Transformer Backbone                   │         │
│   │          (基于 WAN 2.1, 主要冻结/部分微调)                │         │
│   │                                                         │         │
│   │  输入: 视频潜在 Token [B, T', H', W', C]                 │         │
│   │       + 文本 Token [B, L, D]                            │         │
│   │       + 可学习 Token [B, T', N_learn, D] (稀疏任务)      │         │
│   │                                                         │         │
│   │  时间步条件: t = 0 (前馈模式)                            │         │
│   │  输出速度: v = ε - x₀                                   │         │
│   │  取反: -v = x₀ - ε (与目标潜在对齐)                      │         │
│   │                                                         │         │
│   │  关键设计: 直接从最后一层取特征                           │         │
│   │            不对基础模型做架构修改                         │         │
│   └────────────────────────┬────────────────────────────────┘         │
│                            │                                           │
│              ┌─────────────┴─────────────┐                             │
│              ▼                           ▼                             │
│   ┌─────────────────────┐   ┌─────────────────────┐                   │
│   │   稠密任务输出        │   │   稀疏任务输出        │                   │
│   │   (统一 RGB 空间)     │   │   (结构化坐标)       │                   │
│   │                     │   │                     │                   │
│   │   DiT输出 → VAE      │   │   可学习Token → MLP  │                   │
│   │   Decoder → RGB      │   │   → [K] 维预测      │                   │
│   │   [B, T, 3, H, W]    │   │   2D/3D关键点/位姿   │                   │
│   └─────────────────────┘   └─────────────────────┘                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 组件详细设计

#### 2.2.1 VAE 编解码器 (冻结)

| 参数 | 值 | 说明 |
|------|-----|------|
| 空间下采样 | 8× | H, W 各压缩 8 倍 |
| 时间下采样 | 4× | T 压缩 4 倍 (81帧 → 21潜在帧) |
| 训练策略 | **完全冻结** | 保留预训练的视觉压缩先验 |
| 潜在维度 | C=16 (典型值) | 参考 WAN 2.1 配置 |
| 推理优化 | 可卸载到系统内存 | 减少显存占用 |

**关键设计**: VAE 在训练和推理阶段均冻结，不更新权重。这确保了视觉压缩表示的稳定性，避免破坏预训练阶段建立的像素-潜在空间映射。

#### 2.2.2 文本编码器 (冻结)

| 参数 | 值 | 说明 |
|------|-----|------|
| 模型 | T5-XXL / 类似 WAN 的文本编码器 | 与 WAN 2.1 原生兼容 |
| 训练策略 | **完全冻结** | 保留预训练的视觉-语言对齐 |
| 文本嵌入维度 | D=4096 | 典型配置 |
| 最大序列长度 | 512 tokens | |
| 推理优化 | 可卸载到系统内存 | 占约 10GB |

**任务切换机制**: 文本提示直接决定输出模态。例如：
- `"output: depth"` → 深度估计
- `"output: normal"` → 表面法线估计
- `"output: segmentation"` → 前景分割
- `"segment the person wearing red"` → 表达引用分割

#### 2.2.3 DiT Transformer 主干 (选择性微调)

**前馈化改造细节**:

```python
# 核心伪代码
class GenCeptionDiT(nn.Module):
    def __init__(self, pretrained_dit):
        self.dit = pretrained_dit  # 加载 WAN 2.1 DiT
        
    def forward(self, video_latent, text_embed, t=0, learnable_tokens=None):
        """
        Args:
            video_latent: [B, T', H', W', C] 无噪输入视频潜在表示
            text_embed: [B, L, D] 文本嵌入
            t: 0 (固定，表示前馈/终止状态)
            learnable_tokens: [B, T', N, D] 稀疏任务的可学习查询 (可选)
        """
        # 1. 合并所有 token
        tokens = [video_latent]
        if learnable_tokens is not None:
            tokens.append(learnable_tokens)
        tokens.append(text_embed)
        
        # 2. 通过 DiT (时间步 t=0)
        velocity = self.dit(tokens, timestep=t)
        
        # 3. 取反输出 (关键!)
        # 原始 DiT 预测 v = ε - x0
        # 取反后: -v = x0 - ε ≈ x0 (当 ε→0, 即 t→0 时)
        output = -velocity
        
        return output
```

**预训练层复用策略**:

| 训练方案 | 冻结层 | 微调层 | 适用场景 |
|---------|--------|--------|---------|
| 方案 A: 全冻结+LoRA | 全部 DiT 层 | LoRA (to_k, to_v) | 快速验证/低资源 |
| 方案 B: 部分解冻 | 浅层 | 深层 50%+ | 中等资源 |
| 方案 C: 全量微调 | 无 | 全部 | 论文原方案/充足资源 |

论文采用的方案 C（全量微调），但对于开源复现，建议从方案 A 开始验证可行性。

**关键超参数**:

| 参数 | 论文值 | 开源建议值 | 说明 |
|------|--------|-----------|------|
| 模型规模 | 1.3B / 14B | 1.3B 优先 | 14B 需要 8×A100 |
| 输入分辨率 | 480×832 | 480×832 | 与 WAN 2.1 对齐 |
| 帧数 | 81 @ 24fps | 81 @ 24fps | |
| 训练 batch size | 64 | 4-16 (视显存) | |
| 学习率 | 5e-5 | 5e-5 | Adam 优化器 |
| 训练步数 | 15,000 | 5,000-15,000 | |
| Warmup 步数 | 250 | 250-500 | 线性预热 |
| 梯度裁剪阈值 | 配置值 | 1.0 | 关键稳定性技巧 |
| 梯度丢弃阈值 | 更高阈值 | 配置值 | 丢弃异常 batch |

#### 2.2.4 统一任务表示层

**稠密任务: RGB 统一空间**

所有稠密任务输出表示为 3 通道 RGB [0, 1]：

| 任务 | 原始维度 | RGB 编码方式 |
|------|---------|-------------|
| 深度估计 | 1D | 三通道复制 + 非线性映射: d' = clip(α·log(d+1), 0, 1) |
| 表面法线 | 3D | 直接映射三通道 (每个维度对应 R/G/B) |
| 前景分割 | 1D (mask) | 三通道复制 |
| 表达引用分割 | 1D (mask) | 三通道复制 |
| DensePose | 3D (UV+I) | 三通道分别编码 |
| 相机 Raymap | 6D (原点+方向) | **空间划分法**: 中心区域=原点, 外围=方向 |

**深度非线性映射函数** (论文第 3.5 节):

```python
def depth_to_rgb(depth_map, alpha=0.1):
    """
    论文公式: d' = clip(α * log(d + 1), 0, 1)
    alpha 动态调节近场细节与远场结构的关注度
    论文建议通过中位深度归一化消除尺度分歧
    """
    # 1. 中位深度归一化 (消除尺度分歧)
    median_depth = torch.median(depth_map)
    normalized = depth_map / median_depth
    
    # 2. 非线性映射
    mapped = alpha * torch.log(normalized + 1)
    
    # 3. 裁剪到 [0, 1]
    rgb_value = torch.clip(mapped, 0, 1)
    
    # 4. 复制为三通道
    return rgb_value.unsqueeze(1).expand(-1, 3, -1, -1)
```

**相机 Raymap 空间划分** (论文第 3.3 节 + 图5):

```python
def encode_raymap_to_rgb(ray_origins, ray_directions):
    """
    将 6D 相机射线数据 (3D origin + 3D direction) 编码为 3 通道 RGB
    空间划分策略:
    - 中心区域 (如 50% 面积): 编码射线原点 (平移)
    - 外围区域: 编码射线方向 (旋转)
    
    这种表示天然契合视频生成格式，充分利用预训练视觉先验
    """
    H, W = ray_origins.shape[-2:]
    rgb = torch.zeros(3, H, W)
    
    # 中心区域: 射线原点
    center_h, center_w = H // 2, W // 2
    half_h, half_w = center_h // 2, center_w // 2
    rgb[:, center_h-half_h:center_h+half_h, center_w-half_w:center_w+half_w] = \
        ray_origins[:, center_h-half_h:center_h+half_h, center_w-half_w:center_w+half_w]
    
    # 外围区域: 射线方向
    mask = torch.ones(H, W)
    mask[center_h-half_h:center_h+half_h, center_w-half_w:center_w+half_w] = 0
    rgb[:, mask.bool()] = ray_directions[:, mask.bool()]
    
    return rgb
```

**稀疏任务: 可学习 Token 机制**

```python
class SparseTaskTokenAdapter(nn.Module):
    def __init__(self, dim, num_frames_latent, num_tokens_per_frame, output_dim):
        super().__init__()
        # 可学习 token: [num_frames_latent, num_tokens_per_frame, dim]
        self.learnable_tokens = nn.Parameter(
            torch.randn(num_frames_latent, num_tokens_per_frame, dim) * 0.02
        )
        # 时间位置插值 (处理潜在帧数 vs 原始帧数的差异)
        self.temporal_positions = self._create_temporal_positions(num_frames_latent)
        
        # MLP 解码器
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, output_dim)
        )
    
    def forward(self, dit_features):
        """
        Args:
            dit_features: DiT 输出中与 learnable tokens 对应的特征
        Returns:
            predictions: [B, T, num_tokens_per_frame, output_dim]
        """
        # 通过 MLP 解码
        predictions = self.mlp(dit_features)
        return predictions
```

**位置编码设计**:
- 空间位置: 可学习 (因额外 token 没有对应像素位置)
- 时间位置: 使用位置插值 (Position Interpolation)，将 T 个 token 的帧索引降采样到潜在长度 T'
- 应用 DiT 原生 3D RoPE (Rotary Position Embedding)

---

## 3. 数据层设计 (Data Layer)

### 3.1 合成数据生成流水线

由于真实世界视频的真值标注（深度、法线、分割、位姿、关键点）极为稀缺，论文采用**合成数据**作为核心训练来源。开源复现需要构建可扩展的合成数据生成系统。

#### 3.1.1 渲染资产与动作库

| 组件 | 论文配置 | 开源替代方案 |
|------|---------|------------|
| 3D 人物资产 | 800 个 RenderPeople | Mixamo + CC0 人体模型 + 合成角色 |
| 动作库 | 200 种 CMU MoCap | CMU MoCap (公开) + Mixamo 动作 |
| 背景 | 3D 全场景 / HDRI | Poly Haven HDRI + 程序化场景 |
| 渲染引擎 | Blender | Blender (开源) / Unreal Engine |

#### 3.1.2 相机与场景变化策略

```yaml
# 合成数据生成配置
data_generation:
  # 相机参数变化
  camera:
    focal_length_range: [24, 85]  # mm, 等效焦距
    trajectory_types: [static, circular, spline, handheld_noise]
    position_radius_range: [1.5, 5.0]  # 米
    look_at_variation: [center, head, upper_body, random_joint]
  
  # 场景变化
  scene:
    num_backgrounds: 100+  # HDRI 或 3D 场景
    lighting_variations: [day, sunset, indoor, studio]
    weather_effects: [none, fog_light, rain_light]  # 可选
  
  # 人物变化
  character:
    num_identities: 800
    clothing_variations: 5_per_identity
    skin_tone_range: [0.1, 0.9]  # 归一化
    body_shape_variations: [slim, average, athletic, heavy]
  
  # 输出规格
  output:
    resolution: [480, 832]
    fps: 24
    num_frames: 81
    total_videos_target: 7500
```

#### 3.1.3 多通道渲染 (Blender 配置)

```python
# Blender Python API 伪代码
import bpy

def render_multichannel(output_path, frame_start, frame_end):
    scene = bpy.context.scene
    
    # 设置渲染引擎
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    
    # 启用所需渲染通道
    scene.view_layers["ViewLayer"].use_pass_z = True           # 深度
    scene.view_layers["ViewLayer"].use_pass_normal = True      # 法线
    scene.view_layers["ViewLayer"].use_pass_object_index = True # 实例分割
    scene.view_layers["ViewLayer"].use_pass_material_index = True # 语义分割
    
    # 设置输出
    scene.frame_start = frame_start
    scene.frame_end = frame_end
    
    # 渲染 RGB
    scene.render.filepath = f"{output_path}/rgb/"
    bpy.ops.render.render(animation=True)
    
    # 使用 Compositor 输出额外通道...
```

#### 3.1.4 真值标注提取

| 任务 | 真值来源 | 格式 |
|------|---------|------|
| RGB 视频 | Blender Cycles 渲染 | [T, H, W, 3], uint8 |
| 深度图 | Blender Z-pass | [T, H, W, 1], float32 (米) |
| 表面法线 | Blender Normal-pass | [T, H, W, 3], float32 (世界坐标系) |
| 前景分割 | Alpha 通道 / 对象索引 | [T, H, W, 1], binary |
| 2D 关键点 | 骨骼投影到图像平面 | [T, num_joints, 2], float32 (像素) |
| 3D 关键点 | 骨骼世界坐标 | [T, num_joints, 3], float32 (米) |
| 相机位姿 | Blender 相机矩阵 | [T, 4, 4], float32 (SE3) |
| DensePose | UV 展开 + 身体部位索引 | [T, H, W, 3] |

**关键后处理**:
- 深度图: 应用中位深度归一化
- 法线: 转换到相机坐标系并归一化
- 关键点: 处理遮挡 (可见性标志)
- 相机位姿: 转换为射线图表示 (论文图5)

### 3.2 公开数据集整合

| 数据集 | 用途 | 模态 | 规模 | 许可证 |
|--------|------|------|------|--------|
| TartanAir | 补充深度 + 相机轨迹 | 深度, 位姿 | ~1000 seq | BSD |
| Virtual KITTI 2 | 补充深度 | 深度 | ~50 scenes | CC BY-NC-SA |
| MVS Synth | 补充深度 | 深度 | ~100K images | 研究用 |
| MeViS | 表达引用分割训练 | mask, 文本 | 2000 videos | 研究用 |
| Ref-COCO | 表达引用分割训练 | mask, 文本 | 142K expressions | 研究用 |
| YouTube-VOS | 表达引用分割训练 | mask, 文本 | 4000 videos | 研究用 |

### 3.3 数据预处理与缓存

```python
class PreprocessingPipeline:
    """离线预处理: 避免训练时重复计算"""
    
    def __init__(self, vae_encoder, text_encoder):
        self.vae = vae_encoder
        self.text_encoder = text_encoder
    
    def process_video(self, video_path, annotations, task_prompt):
        # 1. 视频 → VAE 潜在表示
        video_tensor = load_video(video_path)  # [T, 3, H, W]
        with torch.no_grad():
            latent = self.vae.encode(video_tensor).latent_dist.sample()
            # latent shape: [T', C, H', W']
        
        # 2. 文本 → 文本嵌入
        with torch.no_grad():
            text_embed = self.text_encoder(task_prompt)
        
        # 3. 目标模态 → VAE 潜在表示 (稠密任务)
        if task_type == 'dense':
            target_rgb = encode_task_to_rgb(annotations, task_type)
            with torch.no_grad():
                target_latent = self.vae.encode(target_rgb).latent_dist.sample()
        
        # 4. 缓存到磁盘
        save_path = f"{cache_dir}/{video_id}_{task_type}.pt"
        torch.save({
            'input_latent': latent,
            'text_embed': text_embed,
            'target_latent': target_latent if task_type == 'dense' else None,
            'target_sparse': annotations if task_type == 'sparse' else None,
        }, save_path)
```

**缓存策略**:
- 预计算并缓存所有 VAE 编码结果
- 预计算并缓存所有文本嵌入
- 训练时直接从缓存加载，避免重复编码
- 预计缓存大小: ~500GB (7500 videos × 81 frames × 多任务)

---

## 4. 训练策略设计 (Training Strategy)

### 4.1 统一损失函数

**核心设计原则**: 所有任务使用完全统一的 L2 (MSE) 损失，不引入任务特定损失。

```python
class UnifiedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 仅使用 L2 损失 — 与 LLM 统一目标类似
    
    def forward(self, predictions, targets, task_type):
        if task_type == 'dense':
            # 稠密任务: L2 在潜在空间
            # predictions, targets: [B, T', C, H', W']
            loss = F.mse_loss(predictions, targets)
        else:
            # 稀疏任务: L2 在输出空间
            # predictions, targets: [B, T, N, K]
            loss = F.mse_loss(predictions, targets)
        return loss
```

**数据层面的定制化替代损失层面的定制化**:

| 传统方法 | GenCeption 替代方案 | 效果 |
|---------|-------------------|------|
| 深度: Scale-Invariant Log Loss | 中位深度归一化 + L2 | 消除尺度分歧 |
| 分割: Cross-Entropy + Dice | 前景 mask RGB 化 + L2 | 统一输出空间 |
| 位姿: 几何损失 (角度+平移) | Raymap RGB 化 + L2 | 统一输出空间 |
| 关键点: Heatmap MSE + OKS | 直接坐标回归 + L2 | 简化训练 |

### 4.2 多任务训练数据混合

```python
class MultiTaskDataset(Dataset):
    """多任务数据混合采样器"""
    
    TASK_MIXING_RATIOS = {
        'depth': 0.20,
        'normal': 0.20,
        'foreground_segmentation': 0.15,
        'referring_segmentation': 0.15,
        'densepose': 0.10,
        'raymap': 0.10,
        'keypoints_2d': 0.05,
        'keypoints_3d': 0.05,
    }
    
    def __getitem__(self, idx):
        # 按混合比例采样任务
        task = random.choices(
            list(self.TASK_MIXING_RATIOS.keys()),
            weights=list(self.TASK_MIXING_RATIOS.values())
        )[0]
        
        # 从对应任务数据集采样
        dataset = self.task_datasets[task]
        sample = dataset[random.randint(0, len(dataset) - 1)]
        
        # 构建统一的训练样本
        return {
            'input_latent': sample['input_latent'],
            'text_embed': sample['text_embed'],
            'target': sample['target'],
            'task': task,
            'text_prompt': f"output: {task}",
        }
```

### 4.3 训练配置

```yaml
# configs/train_genception_base.yaml

model:
  base_model: "Wan-AI/Wan2.1-T2V-1.3B"  # 或 14B
  pretrained_path: null  # 自动从 HuggingFace 下载
  freeze_vae: true
  freeze_text_encoder: true
  freeze_dit: false  # 或 selective
  
  # LoRA 配置 (低资源方案)
  use_lora: false  # 论文全量微调; 设为 true 启用 LoRA
  lora_rank: 64
  lora_alpha: 128
  lora_target_modules: ["to_k", "to_v"]

training:
  output_dir: "./output/genception_base"
  
  # 优化器
  optimizer: "adamw"
  learning_rate: 5.0e-5
  weight_decay: 0.01
  adam_beta1: 0.9
  adam_beta2: 0.999
  adam_epsilon: 1.0e-8
  
  # 调度器
  lr_scheduler: "linear_warmup_cosine"
  warmup_steps: 250
  max_grad_norm: 1.0
  
  # 训练步数
  num_train_steps: 15000
  gradient_accumulation_steps: 1
  
  # 批次
  per_device_batch_size: 4  # 根据显存调整
  num_workers: 4
  
  # 精度
  mixed_precision: "bf16"  # CUDA; MPS 用 fp32
  
  # 稳定性
  gradient_clipping: 1.0
  gradient_dropping_threshold: 10.0  # 丢弃梯度范数超过此值的 batch
  
  # 检查点
  checkpointing_steps: 1000
  save_total_limit: 5
  
  # 日志
  logging_steps: 10
  
  # 种子
  seed: 42

data:
  # 合成数据路径
  synthetic_data_dir: "./data/synthetic"
  # 公开数据路径
  public_data_dir: "./data/public"
  # 缓存路径
  cache_dir: "./data/cache"
  
  # 视频规格
  resolution: [480, 832]
  num_frames: 81
  fps: 24
  
  # 任务混合比例
  task_mixing:
    depth: 0.20
    normal: 0.20
    foreground_segmentation: 0.15
    referring_segmentation: 0.15
    densepose: 0.10
    raymap: 0.10
    keypoints_2d: 0.05
    keypoints_3d: 0.05

evaluation:
  eval_every: 1000
  eval_benchmarks:
    - "sintel_normals"
    - "hi4d_normals"
    - "kitti_depth"
    - "sintel_depth"
    - "eth3d_depth"
    - "goliath_depth"
    - "videomatte"
    - "photomatte"
    - "ref_davis"
    - "mevis"
    - "emdb_keypoints"
    - "sintel_pose"
```

### 4.4 训练稳定性关键技术

| 技术 | 论文应用 | 开源实现建议 | 重要性 |
|------|---------|------------|--------|
| 梯度裁剪 | 是 | `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` | 🔴 必需 |
| 梯度丢弃 | 是 | 丢弃梯度范数 > 10.0 的 batch | 🔴 必需 |
| 线性预热 | 前 250 步 | `get_linear_schedule_with_warmup` | 🟡 推荐 |
| EMA | 可选项 | `EMA(model, decay=0.9999)` | 🟡 推荐 |
| 梯度检查点 | 可选项 | `model.enable_gradient_checkpointing()` | 🟡 显存紧张时 |

---

## 5. 推理与部署设计 (Inference & Deployment)

### 5.1 推理流水线

```python
class GenCeptionInference:
    def __init__(self, model_path, device="cuda"):
        self.vae = load_vae().to(device).eval()
        self.text_encoder = load_text_encoder().to(device).eval()
        self.dit = load_dit(model_path).to(device).eval()
        self.device = device
    
    @torch.no_grad()
    def predict(self, video_path, task_prompt):
        """
        单次前向推理 — 无迭代扩散!
        
        Args:
            video_path: 输入视频路径
            task_prompt: 任务提示, 如 "output: depth"
        Returns:
            根据任务类型返回:
            - 稠密任务: RGB 视频 [T, H, W, 3]
            - 稀疏任务: 结构化预测 [T, N, K]
        """
        # 1. 加载视频
        video = load_video(video_path).to(self.device)
        
        # 2. VAE 编码 (冻结)
        latent = self.vae.encode(video).latent_dist.sample()
        
        # 3. 文本编码 (冻结)
        text_embed = self.text_encoder(task_prompt)
        
        # 4. 判断任务类型
        task_type = self._parse_task(task_prompt)
        
        if task_type == 'dense':
            # 稠密任务: 直接前向
            output_latent = self.dit(latent, text_embed, t=0)
            output_rgb = self.vae.decode(output_latent)
            return output_rgb
        else:
            # 稀疏任务: 注入可学习 token
            learnable_tokens = self.dit.get_learnable_tokens(task_type)
            output_features = self.dit(latent, text_embed, t=0, 
                                       learnable_tokens=learnable_tokens)
            predictions = self.dit.decode_sparse(output_features, task_type)
            return predictions
```

### 5.2 推理性能

| 模型规模 | 硬件 | 时间 (81帧 @ 480×832) | FPS | 显存 |
|---------|------|----------------------|-----|------|
| 1.3B | 单 v6e TPU / A100 | ~6 秒 | 13.6 | 15.3 GB |
| 14B | 单 v6e TPU / A100 | ~10 秒 | 8.0 | 42.8 GB |

**内存优化策略**:
- VAE (0.25GB) + 文本编码器 (10GB) 可卸载到系统内存
- 纯 DiT 推理: 1.3B 需 0.96s, 14B 需 5.11s
- 启用 `torch.compile()` 可再提速 10-20%

### 5.3 部署架构

```
┌──────────────────────────────────────────────────────────────┐
│                    Open-GenCeption API                        │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │ /predict    │  │ /batch      │  │ /health             │  │
│  │ 单视频推理   │  │ 批量推理     │  │ 服务健康检查         │  │
│  └──────┬──────┘  └──────┬──────┘  └─────────────────────┘  │
│         │                │                                   │
│         └────────────────┼─────────────────┐                 │
│                          ▼                 ▼                 │
│              ┌─────────────────────────────────┐             │
│              │      Async Inference Queue      │             │
│              │         (Celery / RQ)           │             │
│              └─────────────────────────────────┘             │
│                          │                                   │
│                          ▼                                   │
│              ┌─────────────────────────────────┐             │
│              │   GenCeption Worker Pool        │             │
│              │   (Docker + GPU)                │             │
│              └─────────────────────────────────┘             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 6. 评估体系设计 (Evaluation)

### 6.1 评估数据集与指标

#### 6.1.1 表面法线估计

| 数据集 | 协议 | 指标 |
|--------|------|------|
| Sintel [9] | 全图评估 | mean AE, median AE, %<11.25°, %<22.5°, %<30° |
| Hi4D [79] | 序列 28, 32, 37; 相机 4; 6 主体; 1195 帧 | 同上 |

#### 6.1.2 深度估计

| 数据集 | 协议 | 指标 |
|--------|------|------|
| KITTI [21] | Eigen split | AbsRel, RMSE, δ1, δ2, δ3 |
| Sintel [9] | 全图 | AbsRel, RMSE |
| ETH3D [57] | 全图 | AbsRel, RMSE |
| Goliath [46] | 12 相机, 16 帧, 4 主体; 面部/上半身/全身 | AbsRel |

#### 6.1.3 前景分割

| 数据集 | 协议 | 指标 |
|--------|------|------|
| VideoMatte [42] | 静态划分复合 | MSE |
| PhotoMatte 85 [42] | 全图 | MSE |
| PPM-100 [31] | 全图 | MSE |

#### 6.1.4 表达引用分割

| 数据集 | 协议 | 指标 |
|--------|------|------|
| Ref-DAVIS [34] | 所有文本查询平均 | J&F (IoU + Contour F-score) |
| MeViS [15] | 所有文本查询平均 | J&F |

#### 6.1.5 3D 关键点估计

| 数据集 | 协议 | 指标 |
|--------|------|------|
| EMDB [29] | 全局轨迹 | MPJPE (Mean Per Joint Position Error) |

#### 6.1.6 相机位姿估计

| 数据集 | 协议 | 指标 |
|--------|------|------|
| Sintel [9] | 全局对齐 | ATE, RPE-T, RPE-R |

### 6.2 与 SOTA 的对比基准

| 任务 | 对比模型 | 论文 GenCeption-L | 开源目标 |
|------|---------|------------------|---------|
| 法线 Sintel | David-Large (0.34B), Sapiens (2B) | mAE=29.3 | mAE < 35 |
| 深度 KITTI | DepthAnything3 (1.15B), D4RT | AbsRel=0.048 | AbsRel < 0.06 |
| 深度 ETH3D | VGGT-Ω (1B) | AbsRel=0.037 | AbsRel < 0.05 |
| 分割 V.Mat | MODNet, David | MSE=0.0027 | MSE < 0.005 |
| 位姿 Sintel | VGGT-Ω | ATE=0.464 | ATE < 0.6 |
| 关键点 EMDB | Genmo, TRAM | MPJPE=10.99 | MPJPE < 15 |
| 引用分割 Ref-D | SAM3 + Gemini | J&F=76.4 | J&F > 70 |

### 6.3 消融实验设计

| 实验 | 目的 | 变量 |
|------|------|------|
| 预训练 vs 从零训练 | 验证预训练重要性 | 冻结层数: 0%, 25%, 50%, 75%, 100% |
| 不同预训练范式对比 | 验证生成式预训练优势 | WAN 2.1 vs V-JEPA vs VideoMAE V2 |
| 单任务 vs 多任务 | 验证联合训练效果 | Specialist 权重 vs Generalist 权重 |
| 数据缩放 | 验证数据效率 | 100, 500, 1000, 2500, 5000, 7500 videos |
| 模型缩放 | 验证模型缩放律 | 1.3B vs 14B |
| 可学习 token 设计 | 验证稀疏任务架构 | 额外 token vs 额外注意力层 |

---

## 7. 实现路线图 (Implementation Roadmap)

### 7.1 阶段划分

```
Phase 0: 基础设施搭建 (Week 1-2)
├── 搭建开发环境 (PyTorch 2.x, diffusers, transformers)
├── 下载并验证 WAN 2.1 权重 (1.3B + 14B)
├── 实现基础模型加载与推理框架
└── 搭建 Blender 合成数据渲染流水线原型

Phase 1: 核心架构验证 (Week 3-5)
├── 实现 VAE + DiT 前馈化改造
├── 验证 t=0 修正流输出的取反逻辑
├── 实现统一 RGB 输出空间 (深度/法线/分割)
├── 生成第一批合成数据 (100 视频)
└── 深度估计单任务训练与验证

Phase 2: 多任务扩展 (Week 6-10)
├── 扩展至 6+ 任务 (深度/法线/分割/DensePose/Raymap/关键点)
├── 实现可学习 token 机制 (稀疏任务)
├── 大规模合成数据生成 (7500 视频)
├── 公开数据集整合
└── 多任务联合训练

Phase 3: 评估与优化 (Week 11-14)
├── 在 11 个 benchmark 上执行评估
├── 消融实验 (预训练、数据缩放、模型缩放)
├── 推理优化 (量化、编译、批处理)
└── 涌现行为验证 (sim-to-real, OOD 泛化)

Phase 4: 发布与文档 (Week 15-16)
├── 模型权重发布 (HuggingFace)
├── 训练代码开源 (GitHub)
├── 推理 API 与演示
├── 技术报告撰写
└── 社区文档完善
```

### 7.2 关键里程碑

| 里程碑 | 目标 | 验收标准 | 预计时间 |
|--------|------|---------|---------|
| M1: 架构闭环 | 前馈 DiT 可输出可解码的 RGB | VAE decode 后 PSNR > 30dB | Week 3 |
| M2: 深度单任务 | 深度估计可训练且收敛 | 合成数据 loss < 0.3 | Week 5 |
| M3: 多任务训练 | 6 任务联合训练无发散 | 各任务 loss 均稳定下降 | Week 8 |
| M4: 基准达标 | 至少 3 个 benchmark 达到可接受下限 | 见 6.2 表 | Week 12 |
| M5: 涌现验证 | sim-to-real 和 OOD 泛化可观测 | 定性结果与论文图 10 类似 | Week 14 |
| M6: 开源发布 | 完整可复现的训练+推理代码 | 社区可在 24h 内复现训练 | Week 16 |

### 7.3 代码仓库结构

```
open-genception/
├── README.md
├── LICENSE
├── setup.py / pyproject.toml
├── requirements.txt
│
├── configs/                          # 训练配置
│   ├── train_1.3b.yaml
│   ├── train_14b.yaml
│   ├── train_lora.yaml
│   └── data_generation.yaml
│
├── src/
│   ├── genception/
│   │   ├── __init__.py
│   │   ├── model/
│   │   │   ├── __init__.py
│   │   │   ├── dit_forward.py        # 前馈化 DiT
│   │   │   ├── sparse_adapter.py     # 可学习 token + MLP
│   │   │   └── genception.py         # 主模型封装
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── synthetic_generator/  # Blender 渲染脚本
│   │   │   │   ├── blender_render.py
│   │   │   │   ├── asset_loader.py
│   │   │   │   └── camera_controller.py
│   │   │   ├── preprocess.py         # 离线预处理
│   │   │   ├── dataset.py            # PyTorch Dataset
│   │   │   └── task_mixing.py        # 多任务混合采样
│   │   ├── training/
│   │   │   ├── __init__.py
│   │   │   ├── trainer.py            # 训练循环
│   │   │   ├── losses.py             # 统一损失
│   │   │   └── utils.py              # 梯度裁剪/丢弃/EMA
│   │   ├── inference/
│   │   │   ├── __init__.py
│   │   │   ├── pipeline.py           # 推理流水线
│   │   │   └── api.py                # 部署 API
│   │   └── evaluation/
│   │       ├── __init__.py
│   │       ├── depth_metrics.py
│   │       ├── normal_metrics.py
│   │       ├── segmentation_metrics.py
│   │       ├── pose_metrics.py
│   │       └── benchmark_runner.py
│   └── scripts/
│       ├── train.py                  # 训练入口
│       ├── evaluate.py               # 评估入口
│       ├── generate_synthetic_data.py # 合成数据生成
│       ├── preprocess_data.py         # 数据预处理
│       └── demo.py                    # 交互式演示
│
├── data/                             # 数据目录 (gitignore)
│   ├── synthetic/
│   ├── public/
│   └── cache/
│
├── outputs/                          # 训练输出 (gitignore)
│   └── checkpoints/
│
├── tests/                            # 单元测试
│   ├── test_model.py
│   ├── test_data.py
│   └── test_training.py
│
└── docs/                             # 文档
    ├── ARCHITECTURE.md
    ├── TRAINING.md
    ├── EVALUATION.md
    └── DATA_GENERATION.md
```

---

## 8. 技术栈与依赖 (Tech Stack)

### 8.1 核心依赖

| 组件 | 推荐版本 | 用途 |
|------|---------|------|
| Python | 3.10+ | 运行时 |
| PyTorch | 2.2+ | 深度学习框架 |
| torchvision | 0.17+ | 视频/图像处理 |
| diffusers | 0.30+ | 扩散模型基座 (WAN 2.1) |
| transformers | 4.40+ | 文本编码器 |
| accelerate | 0.30+ | 分布式训练 |
| peft | 0.11+ | LoRA (可选) |
| xformers | 0.0.25+ | 注意力优化 |
| decord / av | 最新 | 视频解码 |
| opencv-python | 4.x | 图像处理 |
| numpy | 1.24+ | 数值计算 |
| pillow | 10+ | 图像 I/O |
| wandb / tensorboard | 最新 | 实验跟踪 |
| hydra-core | 1.3+ | 配置管理 |
| einops | 最新 | 张量操作 |

### 8.2 合成数据生成依赖

| 组件 | 版本 | 用途 |
|------|------|------|
| Blender | 4.0+ | 渲染引擎 |
| bpy | 随 Blender | Python API |
| trimesh | 最新 | 3D 网格处理 |
| pyrender | 最新 | 轻量渲染 (替代方案) |
| opendr | 最新 | 可微渲染 |

### 8.3 硬件需求

| 场景 | 最低配置 | 推荐配置 | 论文配置 |
|------|---------|---------|---------|
| 开发/调试 | 1× RTX 4090 (24GB) | 1× A100 40GB | 1× v6e TPU |
| 1.3B 训练 | 1× A100 40GB | 1× A100 80GB | 256× v6e TPU |
| 14B 训练 | 4× A100 80GB | 8× A100 80GB | 256× v6e TPU |
| 推理 | 1× RTX 3090 | 1× A100 40GB | 1× v6e TPU |

---

## 9. 风险与缓解策略 (Risk & Mitigation)

### 9.1 技术风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|---------|
| **WAN 2.1 开源权重与论文内部版本差异** | 高 | 高 | 明确基准差异; 从 1.3B 验证可行性; 文档记录所有偏差 |
| **前馈化改造后输出质量不足** | 中 | 致命 | 逐步验证: 从 1-step 到 4-step 再到 pure feedforward; 保留扩散采样作为 fallback |
| **合成数据与真实世界 domain gap** | 高 | 高 | 这是论文核心贡献点 (sim-to-real 已验证); 若失败则聚焦 domain adaptation 技术 |
| **多任务联合训练发散** | 中 | 高 | 先训练 Specialist 模型验证各任务可行性; 逐步合并; 监控梯度冲突 |
| **可学习 token 破坏预训练注意力** | 中 | 高 | 对比实验: 额外 token vs 额外注意力层; 冻结更多 DiT 层 |
| **稀疏任务 (关键点) 性能不足** | 中 | 中 | 论文本身报告了联合训练对关键点的损害; 可保留独立专家模型 |

### 9.2 资源风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|---------|
| **合成数据渲染成本过高** | 高 | 高 | 先渲染 1000 视频验证; 使用云渲染农场; 考虑用现有数据集替代 |
| **训练计算资源不足** | 高 | 高 | LoRA 方案作为 fallback; 寻求云计算资助 (AWS/Azure/GCP); 社区众筹 |
| **存储空间不足 (缓存)** | 中 | 中 | 动态加载 VAE 编码 (训练时实时); 压缩缓存; 使用对象存储 |

### 9.3 外部依赖风险

| 风险 | 可能性 | 影响 | 缓解策略 |
|------|--------|------|---------|
| **WAN 2.1 权重许可变更** | 低 | 致命 | 遵守当前许可; 本地备份权重; 备选基座: CogVideo, Mochi, CausVid |
| **diffusers 库 API 变更** | 中 | 中 | 锁定版本; 封装适配层; CI 集成兼容性测试 |
| **Blender 渲染 pipeline 不稳定** | 中 | 中 | 使用 Docker 容器化; 完整渲染日志; 断点续传 |

### 9.4 关键决策点

| 决策点 | 触发条件 | 选项 | 默认选择 |
|--------|---------|------|---------|
| DG-1: 基座选择 | Phase 0 结束 | WAN 2.1 / CogVideo / 其他 | WAN 2.1 (论文原基座) |
| DG-2: 训练策略 | M1 验证失败 | 全量微调 / LoRA / 部分解冻 | LoRA (资源受限时) |
| DG-3: 数据规模 | M2 收敛但性能不足 | 增加合成数据 / 引入真实数据 / 调参 | 增加合成数据至 7500 |
| DG-4: 多任务策略 | M3 发散 | 联合训练 / 独立专家 / 任务分组 | 独立专家 (保守) |
| DG-5: 发布策略 | M4 评估完成 | 全量开源 / 仅权重 / 仅推理 API | 全量开源 |

---

## 10. 与 Open-Vision-Banana 的集成方案

### 10.1 架构定位

Open-GenCeption 可作为 open-vision-banana 的 **视频感知扩展模块**，与其现有的图像感知能力形成互补：

```
┌──────────────────────────────────────────────────────────────┐
│                   open-vision-banana                         │
│                   (统一视觉理解平台)                          │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────┐  ┌─────────────────────────┐   │
│  │   图像感知模块           │  │   视频感知模块 (新增)    │   │
│  │   (基于 FLUX.1)         │  │   (基于 WAN 2.1)        │   │
│  │                         │  │                         │   │
│  │  - 深度估计             │  │  - 深度估计 (时序一致)   │   │
│  │  - 表面法线             │  │  - 表面法线 (时序一致)   │   │
│  │  - 语义分割             │  │  - 前景分割             │   │
│  │  - 实例分割             │  │  - 表达引用分割         │   │
│  │  - 指代分割             │  │  - DensePose            │   │
│  │                         │  │  - 2D/3D 关键点         │   │
│  │                         │  │  - 相机位姿估计         │   │
│  └─────────────────────────┘  └─────────────────────────┘   │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐│
│  │              统一接口层 (Unified Interface)               ││
│  │         输入: 图像或视频 + 文本指令                        ││
│  │         输出: 根据任务类型自动路由到对应模块                ││
│  └─────────────────────────────────────────────────────────┘│
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### 10.2 复用组件

| 组件 | 复用来源 | 适配工作 |
|------|---------|---------|
| 训练基础设施 | open-vision-banana `TrainerInterface` | 扩展支持视频数据加载器 |
| 配置系统 | Hydra configs | 新增视频相关配置项 |
| 日志系统 | WandB/TensorBoard | 新增视频可视化 (GIF 序列) |
| LoRA 实现 | PEFT 集成 | 适配 DiT 注意力层名称 |
| 评估框架 | 现有 metrics | 新增时序一致性指标 |

### 10.3 独立扩展路径

如果作为独立项目，Open-GenCeption 可以：
- 复用 open-vision-banana 的训练抽象层和配置范式
- 引入视频特有的数据增强 (时序裁剪、帧率变化)
- 专注视频生成模型的感知再利用这一独特方向

---

## 11. 附录

### 附录 A: 核心公式速查

**1. 修正流速度预测 (DiT 原始训练目标)**:
```
v = ε - x₀
```
其中 ε 是噪声，x₀ 是干净潜在表示。

**2. 前馈化取反 (GenCeption 关键)**:
```
output = -v = x₀ - ε ≈ x₀  (当 t→0, ε→0)
```

**3. 深度非线性映射**:
```
d' = clip(α · log(d + 1), 0, 1)
```

**4. 深度中位归一化 (消除尺度)**:
```
d_normalized = d / median(d)
```

**5. 相机 Raymap 编码**:
```
中心区域 (50%): R_origins = [O_x, O_y, O_z]
外围区域 (50%): R_directions = [D_x, D_y, D_z]
合并: Raymap = SpatialCombine(R_origins, R_directions)
```

### 附录 B: 论文关键图表索引

| 图号 | 内容 | PRD 对应章节 |
|------|------|------------|
| 图 1 | 方法论 + 范式转变 | 1.2, 2.1 |
| 图 2 | SOTA 能力 + 数据效率 | 6.2 |
| 图 3 | 能力展示 (多任务定性) | 1.1 |
| 图 4 | 架构概览 | 2.1, 2.2 |
| 图 5 | Raymap RGB 编码 | 2.2.4 |
| 图 6 | 深度/法线演示 | 6.1 |
| 图 7 | 引用分割定性 | 6.1 |
| 图 8 | 3D 姿态估计 | 6.1 |
| 图 9 | 预训练层迁移消融 | 6.3 |
| 图 10 | 涌现行为 (多实例 + OOD) | 6.3, 9.1 |

### 附录 C: 术语表

| 术语 | 说明 |
|------|------|
| DiT | Diffusion Transformer，扩散 Transformer |
| Rectified Flow | 修正流，一种无需模拟的流匹配方法 |
| VAE | Variational Autoencoder，变分自编码器 |
| LoRA | Low-Rank Adaptation，低秩适配 |
| RoPE | Rotary Position Embedding，旋转位置编码 |
| Sim-to-Real | 从合成数据到真实世界的迁移 |
| OOD | Out-of-Distribution，分布外 |
| ATE | Absolute Trajectory Error，绝对轨迹误差 |
| RPE | Relative Pose Error，相对位姿误差 |
| MPJPE | Mean Per Joint Position Error，关节位置误差均值 |
| J&F | Jaccard & F-score，分割评估指标 |

---

*本文档由基于 arXiv:2607.09024 论文分析生成，旨在指导 Open-GenCeption 的开源复现与实现。*
