# Open-Vision-Banana Inference Pipeline 深度代码审计报告

> **审计范围**: `src/open_vision_banana/inference/` 全部 7 个模块  
> **审计时间**: 2025-07-07  
> **审计工具**: mypy 1.x, pytest 9.x + coverage, ruff 0.15.x  
> **Python 版本**: 3.12.9

---

## 模块概览

| 文件 | 行数 | 测试覆盖率 | mypy 状态 | ruff 状态 |
|------|------|-----------|-----------|-----------|
| `__init__.py` | 35 | **100%** | ✅ | ✅ |
| `pipeline.py` | 279 | **32%** | ❌ 3 errors | ✅ |
| `generator.py` | 252 | **34%** | ✅ | ✅ |
| `decoder.py` | 95 | **79%** | ✅ | ✅ |
| `optimization.py` | 95 | **83%** | ✅ | ✅ |
| `checkpoint.py` | 69 | **75%** | ✅ | ✅ |
| `prompt.py` | 62 | **90%** | ✅ | ✅ |
| **合计** | **887** | **51%** | ❌ | ✅ |

---

## 1. 当前状态

### 1.1 架构设计

`inference/` 模块采用清晰的分层架构：

- **`VisionBanana` (pipeline.py)**: 高层统一 API，封装任务分发、批处理、瓦片推理
- **`BaseGenerator` (generator.py)**: FLUX.1 + LoRA 包装器，处理文本编码、图像生成
- **`VisualizationDecoder` (decoder.py)**: RGB → 任务输出的解码层
- **`PromptBuilder` (prompt.py)**: 任务特定的提示词构建
- **`InferenceOptions` (optimization.py)**: 内存/速度优化配置
- **`checkpoint.py`**: LoRA 检查点解析与下载

**设计亮点**：
- KV-cache 复用（`_PromptEmbedCache` LRU 缓存）避免重复文本编码
- 延迟加载双管道（t2i / i2i）节省内存
- 瓦片推理支持超大分辨率图像
- 流式推理 callback 机制支持渐进式 UI

### 1.2 测试现状

| 测试文件 | 相关测试数 | 覆盖模块 | 关键测试 |
|----------|-----------|----------|----------|
| `test_integration.py` | 6 | pipeline, decoder | VisionBanana mock 推理流程 |
| `test_checkpoint.py` | 11 | checkpoint | 注册表解析、HF 前缀处理 |
| `test_optimizations_v021.py` | 22 | generator, optimization | PromptEmbedCache LRU、流式推理、瓦片回退 |
| `test_v032_features.py` | 5 | optimization | INT8 量化、VRAM 估算 |
| **合计** | **~44** | | |

**集成测试（test_integration.py）覆盖**：
- ✅ `predict()` depth/normal/referring/semantic 任务
- ✅ `predict_batch()` 结果拆分
- ✅ 不支持的任务抛出 ValueError
- ✅ `decode_referring()` 亮度/暗度/形状测试
- ✅ `_output_to_tensor()` numpy/PIL/tensor 多格式处理
- ✅ `InferenceOptions` dtype 同步

### 1.3 类型安全

- 全面使用 `from __future__ import annotations`（PEP 563）
- 所有公共 API 均有类型注解
- 使用 `|` 联合类型语法（Python 3.10+）
- **但存在 3 个 mypy 错误**（详见 P0 章节）

---

## 2. 发现的问题

### 🔴 P0 — 阻塞生产部署

#### P0-1: `pipeline.py:270` 变量重定义（类型安全违规）

```python
# Line 252
item: dict = {}
for key, val in result.items():
    ...
    item[key] = val

# Line 270（重复逻辑块）
item: dict = {}  # noqa: F811  ← mypy [no-redef]
```

**影响**: 虽然运行时无问题，但 mypy 报错阻止 CI 通过严格类型检查。`# noqa: F811` 仅抑制 flake8，不抑制 mypy。

**根因**: `predict_batch()` 中两个分支（带 prompts / 不带 prompts）的 per-image 拆分逻辑完全复制粘贴。

---

#### P0-2: `pipeline.py:642-676` `export_torchscript()` 中类型不兼容赋值

```python
# Line 642
if task == "depth":
    from ..core.depth_mapper import DepthRGBMapper
    mapper = DepthRGBMapper()           # mapper 被推断为 DepthRGBMapper
    class DepthDecoderScript(...): ...
    script_model = DepthDecoderScript(...)  # script_model 被推断为 DepthDecoderScript

# Line 660
elif task == "normal":
    from ..core.normal_mapper import NormalRGBMapper
    mapper = NormalRGBMapper()          # ❌ [assignment] 与 DepthRGBMapper 不兼容
    class NormalDecoderScript(...): ...
    script_model = NormalDecoderScript(...)  # ❌ [assignment] 与 DepthDecoderScript 不兼容
```

**影响**: 严重类型安全违规。条件分支内重新赋值不同类型的变量，破坏静态类型分析。

**根因**: 在单个方法体内通过条件分支切换类型，未使用 `TypeVar` 或提取为独立方法。

---

### 🟡 P1 — 高优先级（影响可靠性与可维护性）

#### P1-1: 核心推理模块测试覆盖率严重不足

| 模块 | 覆盖率 | 未覆盖关键路径 |
|------|--------|-------------|
| `pipeline.py` | **32%** | `from_pretrained()` 工厂方法、`predict()` 完整生成流程、`predict_batch(prompts=...)` 分支、`predict_tiled()` 瓦片逻辑、`predict_streaming()` callback、`_predict_tiled_discrete()`、`_build_prompt()` 全部 10 个任务分支、`_decode_output()` 全部 10 个任务分支、`export_torchscript()` |
| `generator.py` | **34%** | `_load_model()` FLUX 管道加载、`_get_i2i_pipe()` 构建、`encode_prompts()` 批量编码与缓存、`generate()` 主生成逻辑（img2img/t2i 分支）、`_to_pil_list()`、`save_lora()`、`unload_lora()`、`enable_cpu_offload()`、`_load_and_merge_lora()` 手动 LoRA 合并（~90 行核心代码） |

**影响**: 核心推理路径缺乏回归测试，重构风险极高。特别是 `_load_and_merge_lora()` 是 LoRA 权重手动合并的 fallback 路径，涉及 safetensors 解析和权重注入，一旦出错将导致模型输出静默损坏。

**建议**: 
- 使用 `unittest.mock.patch` 模拟 `diffusers.FluxPipeline` 和 `diffusers.FluxImg2ImgPipeline`
- 对 `_load_and_merge_lora()` 构建独立的单元测试，使用小型随机 safetensors 文件验证权重注入正确性
- 对 `predict_tiled()` 使用固定种子的小张量测试瓦片拼接结果

---

#### P1-2: `generator.py:135-136` 使用 `Any` 逃避类型检查

```python
self._t2i_pipe: Any = None  # FluxPipeline (text → image)
self._i2i_pipe: Any = None  # FluxImg2ImgPipeline (image + text → image)
```

**影响**: 类型系统无法追踪管道对象的属性和方法调用，失去静态类型保护。

**建议**: 使用 `TYPE_CHECKING` 条件导入 diffusers 类型：

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from diffusers import FluxPipeline, FluxImg2ImgPipeline

class BaseGenerator:
    def __init__(...):
        self._t2i_pipe: FluxPipeline | None = None
        self._i2i_pipe: FluxImg2ImgPipeline | None = None
```

---

#### P1-3: `generator.py:183-189` LoRA 加载 fallback 缺乏双重保护

```python
try:
    pipe.load_lora_weights(self._resolved_lora_path)
except Exception as e:
    logger.warning("pipe.load_lora_weights failed (%s); falling back...", e)
    self._load_and_merge_lora(pipe, self._resolved_lora_path)
```

**影响**: 如果 `_load_and_merge_lora()` 也失败，异常将直接冒泡，导致模型加载失败且错误信息可能误导用户（会同时看到 warning 和未处理的 exception）。

**建议**: 添加双重 try-except，并在最终失败时抛出明确的 `RuntimeError`：

```python
except Exception as e2:
    raise RuntimeError(
        f"LoRA loading failed: primary ({e}) and fallback ({e2}). "
        f"Path: {self._resolved_lora_path}"
    ) from e2
```

---

#### P1-4: `pipeline.py:393-397` `eval()` 异常处理过于宽泛

```python
try:
    pipe.eval()
except AttributeError as e:
    logger.warning("Failed to call eval() on pipeline: %s", e)
```

**影响**: 仅捕获 `AttributeError`，但 `pipe.eval()` 可能抛出 `RuntimeError` 或其他异常（如 CUDA 错误），导致调用者无法预期地崩溃。

**建议**: 扩大异常捕获范围，或明确文档化仅保证 `AttributeError` 被处理：

```python
except Exception as e:
    logger.warning("Failed to call eval() on pipeline: %s", e)
```

---

#### P1-5: `pipeline.py:406-407` 直接访问私有属性破坏封装

```python
def clear_cache(self) -> None:
    """Clear the prompt embedding cache and generator KV-cache."""
    self.generator._prompt_cache.clear()
```

**影响**: `_prompt_cache` 是 `BaseGenerator` 的私有属性（单下划线约定）。`VisionBanana` 作为调用方不应直接操作。

**建议**: 在 `BaseGenerator` 中添加 `clear_prompt_cache()` 公共方法，由 `VisionBanana.clear_cache()` 调用。

---

#### P1-6: `generator.py:399` 函数内重复 import

```python
@staticmethod
def _output_to_tensor(output) -> torch.Tensor:
    images = output.images
    import numpy as np  # ← 每次调用都重新导入
```

**影响**: Python import 有模块缓存，实际性能影响极小，但违反 PEP 8（imports 应放在模块顶部）。

---

### 🟢 P2 — 中等优先级（代码质量与优化）

#### P2-1: `pipeline.py:250-261` 与 `268-278` 代码重复

两个分支中 per-image 结果拆分逻辑完全复制，仅变量名不同。

**建议**: 提取为 `_split_batch_result(result: dict, batch_size: int) -> list[dict]` 辅助方法。

---

#### P2-2: `pipeline.py:623-683` `export_torchscript()` 方法过长且职责不单一

方法包含两个几乎相同的内部类定义（`DepthDecoderScript` / `NormalDecoderScript`），违反 SRP。

**建议**: 提取为 `DepthTorchScriptExporter` 和 `NormalTorchScriptExporter` 两个独立类，或使用工厂模式。

---

#### P2-3: `prompt.py` 模板系统扩展性不足

`PromptBuilder` 使用硬编码的类属性模板，添加新任务需要修改类定义。

**建议**: 考虑将模板外置到 YAML/JSON 配置文件，支持运行时热加载。

---

#### P2-4: `checkpoint.py:19` 硬编码的注册表路径

```python
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[3] / "configs" / "checkpoint_registry.yaml"
```

**影响**: 如果包被安装到 site-packages（而非 editable 模式），`parents[3]` 可能指向不可预期的位置。

**建议**: 使用 `importlib.resources` 或 `pkgutil` 访问包内资源文件。

---

#### P2-5: `optimization.py:87-131` `quantize_mapper_int8()` 异常处理过于宽泛

```python
except (ImportError, AttributeError, RuntimeError, NotImplementedError) as e:
```

虽然这是有意的 fallback 设计，但捕获 `RuntimeError` 可能掩盖真正的运行时错误（如内存不足）。

**建议**: 细化异常类型，或在日志中区分 "torchao 不可用" 和 "torchao 运行时失败"。

---

#### P2-6: `generator.py:228-284` `encode_prompts()` 返回类型不够精确

```python
def encode_prompts(...) -> dict[str, torch.Tensor | None]:
```

返回值包含 `prompt_embeds` 和 `pooled_prompt_embeds`，但调用方需要通过字符串键访问，缺乏编译时检查。

**建议**: 使用 `TypedDict`：

```python
class PromptEmbedDict(TypedDict):
    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor | None
```

---

#### P2-7: `pipeline.py` 返回值使用普通 `dict`

`predict()` 和 `predict_batch()` 返回 `dict`，各任务返回的键不统一（depth → `"depth"`, semantic → `"mask"`, edge → `"edge_mask"`），调用方无法静态验证。

**建议**: 为每个任务定义 `TypedDict` 返回类型，或使用 `dataclass`。

---

## 3. 修复建议

### 立即执行（P0 阻断）

1. **修复变量重定义** (`pipeline.py:270`):
   ```python
   # 提取辅助方法
   def _split_batch_result(self, result: dict, batch_size: int) -> list[dict]:
       per_image = []
       for i in range(batch_size):
           item: dict = {}
           for key, val in result.items():
               if isinstance(val, torch.Tensor) and val.dim() > 0 and val.shape[0] == batch_size:
                   item[key] = val[i:i+1]
               elif isinstance(val, list) and len(val) == batch_size:
                   item[key] = val[i]
               else:
                   item[key] = val
           per_image.append(item)
       return per_image
   ```

2. **修复 `export_torchscript()` 类型问题** (`pipeline.py:623-683`):
   提取为独立函数或使用 `Protocol` 定义统一接口：
   ```python
   from typing import Protocol
   
   class _TorchScriptExporter(Protocol):
       def export(self, output_path: Path, input_shape: tuple) -> None: ...
   
   class _DepthTorchScriptExporter:
       def export(self, output_path: Path, input_shape: tuple) -> None:
           from ..core.depth_mapper import DepthRGBMapper
           mapper = DepthRGBMapper()
           ...
   
   class _NormalTorchScriptExporter:
       def export(self, output_path: Path, input_shape: tuple) -> None:
           from ..core.normal_mapper import NormalRGBMapper
           mapper = NormalRGBMapper()
           ...
   ```

### 短期执行（P1 高优先级）

3. **提升 `generator.py` 测试覆盖率至 70%+**:
   - 使用 `unittest.mock.MagicMock` 模拟 `diffusers` 管道
   - 为 `_load_and_merge_lora()` 编写单元测试（创建临时 safetensors 文件）
   - 测试 `encode_prompts()` 的缓存命中/未命中/驱逐逻辑

4. **提升 `pipeline.py` 测试覆盖率至 60%+**:
   - 测试 `predict_batch(prompts=[...])` 分支
   - 测试 `predict_tiled()` 的瓦片拼接逻辑（使用固定种子小输入）
   - 测试 `_predict_tiled_discrete()` 的投票逻辑
   - 测试 `export_torchscript()` 的文件写入

5. **替换 `Any` 类型** (`generator.py:135-136`): 使用 `TYPE_CHECKING` 条件导入 diffusers 类型。

6. **加强 LoRA 加载错误处理** (`generator.py:183-189`): 添加双重异常捕获和明确的错误消息。

7. **封装 `_prompt_cache` 访问** (`generator.py`): 添加 `clear_prompt_cache()` 公共方法。

### 中期执行（P2 优化）

8. **使用 `TypedDict` / `dataclass` 精化返回类型**: 为每个任务的 predict 输出定义结构化类型。
9. **将 numpy import 移至模块顶部** (`generator.py:399`)。
10. **使用 `importlib.resources` 替代硬编码路径** (`checkpoint.py:19`)。
11. **提取 per-image 拆分逻辑为辅助方法** 消除代码重复。
12. **外置 PromptBuilder 模板** 到配置文件，提升可扩展性。

---

## 4. 生产就绪度评分

### 评分维度

| 维度 | 权重 | 得分 | 说明 |
|------|------|------|------|
| **代码结构** | 15% | **8/10** | 清晰分层，职责分离良好，命名规范 |
| **类型安全** | 20% | **5/10** | 全面注解但存在 3 个 mypy 错误 + Any 逃避 |
| **测试覆盖** | 25% | **4/10** | 核心模块 32-34%，远低于生产级 80% 标准 |
| **错误处理** | 15% | **6/10** | 基本完善但存在过于宽泛/狭窄的捕获 |
| **文档质量** | 10% | **8/10** | 文档字符串全面，注释充分 |
| **lint/format** | 10% | **9/10** | ruff 全部通过，代码风格一致 |
| **可维护性** | 5% | **6/10** | 代码重复，部分方法过长 |

### 最终评分

```
总分 = 8×0.15 + 5×0.20 + 4×0.25 + 6×0.15 + 8×0.10 + 9×0.10 + 6×0.05
     = 1.20 + 1.00 + 1.00 + 0.90 + 0.80 + 0.90 + 0.30
     = 6.10
```

## **生产就绪度评分: 6.1 / 10**

### 评分解读

| 区间 | 状态 | 行动建议 |
|------|------|----------|
| 8.0–10.0 | 🟢 生产就绪 | 可直接部署 |
| 6.0–7.9 | 🟡 有条件就绪 | 修复 P0/P1 后可进入 staging |
| 4.0–5.9 | 🟠 预生产 | 需大量测试补充和架构调整 |
| 0–3.9 | 🔴 不建议 | 重大重构 needed |

**当前状态: 🟡 有条件就绪**

`inference_pipeline` 模块架构设计优秀，API 设计合理，文档完善。但核心推理路径（`generator.py` 和 `pipeline.py`）的测试覆盖率严重不足（32-34%），且存在 3 个 mypy 类型错误。在修复 P0 问题并将核心模块测试覆盖率提升至 70%+ 后，可达到生产部署标准。

---

## 附录 A: 测试覆盖详情

### 未覆盖行号分布

| 模块 | 未覆盖行数 | 主要未覆盖区域 |
|------|-----------|-------------|
| `pipeline.py` | 191 | `from_pretrained`(95-109), `predict_batch` prompts 分支(215-263), `_build_prompt` 全部分支(311-326), `_decode_output` 大部分分支(344-374), `eval`(389-398), `save`(402-403), `clear_cache`(407), `predict_tiled`(483, 491-540), `_predict_tiled_discrete`(551-621), `export_torchscript`(639-683) |
| `generator.py` | 167 | `_prompt_cache_key`(33-35), `_tensor_to_pil`(66-72), `_pil_to_tensor`(75-84), `_resolve_device`(151-159), `_load_model`(167-198), `_get_i2i_pipe`(202-226), `encode_prompts`(238-281), `generate`(334-377), `_to_pil_list`(385-393), `_output_to_tensor`(396-427), `save_lora`(438-443), `unload_lora`(445-451), `_load_and_merge_lora`(465-546), `enable_cpu_offload`(550-553) |
| `decoder.py` | 20 | `decode_semantic`(86-89), `decode_instance`(101-104), `decode_edge` threshold≠0.5 分支(194-199), `decode_flow` max_flow≠20 分支(220-222), `decode_keypoint`(244-245), `decode_object_detection`(269-270), `decode_depth_edge`(293-294) |
| `optimization.py` | 16 | `model_cpu_offload`(44-45), `sequential_cpu_offload`(48-49), `vae_tiling`(56-57), `torch_compile`(64-72), `torchao fallback`(116-118), `estimate_vram_savings` 部分分支(144, 146, 154) |
| `checkpoint.py` | 17 | `load_checkpoint_registry` warning(28-29), `resolve_lora_repo_id` placeholder(78) + repo_id 缺失(84), `download_lora_from_hub` 完整函数(107-130), `list_registry_checkpoints` 完整函数(182-193) |
| `prompt.py` | 6 | `build_semantic_prompt` template_idx=1/2 分支(82, 129, 134, 139, 144, 149) |

### 附录 B: mypy 错误详情

```
src/open_vision_banana/inference/pipeline.py:270: error: Name "item" already
    defined on line 252  [no-redef]
src/open_vision_banana/inference/pipeline.py:663: error: Incompatible types in
    assignment (expression has type "NormalRGBMapper", variable has type
    "DepthRGBMapper")  [assignment]
src/open_vision_banana/inference/pipeline.py:673: error: Incompatible types in
    assignment (expression has type "NormalDecoderScript", variable has type
    "DepthDecoderScript")  [assignment]
```

---

*报告生成完成。建议优先修复 P0 问题，随后按 P1 列表补充测试。*
