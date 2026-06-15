#!/usr/bin/env python3
"""批量改写 SparkLM 跨包导入：旧顶层包 -> sparklm 子包。

映射:
  dataset    -> sparklm.data
  trainer    -> sparklm.training
  evaluation -> sparklm.eval
  scripts    -> sparklm.cli

处理三种形态(词边界精确):
  from <pkg>.xxx      -> from sparklm.<new>.xxx
  from <pkg> import   -> from sparklm.<new> import
  import <pkg>.xxx    -> import sparklm.<new>.xxx
  import <pkg> as     -> import sparklm.<new> as
还处理字符串内 monkeypatch/import_module 的 model. 旧引用 -> sparklm.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TARGET_DIRS = [ROOT / "src" / "sparklm", ROOT / "tests"]

MAP = {
    "dataset": "sparklm.data",
    "trainer": "sparklm.training",
    "evaluation": "sparklm.eval",
    "scripts": "sparklm.cli",
}

# model. 旧命名空间 -> sparklm. (用于 monkeypatch / import_module 字符串与真实 import)
MODEL_MODULE_MAP = {
    "model.moe_v4": "sparklm.moe_v4",
    "model.model_sparklm": "sparklm.model_sparklm",
    "model.components": "sparklm.components",
    "model.config": "sparklm.config",
    "model.kv_cache": "sparklm.kv_cache",
    "model.attention_v4": "sparklm.attention_v4",
    "model.kv_cache_quant": "sparklm.kv_cache_quant",
    "model.tokenizer": "sparklm.tokenizer",
}

changed_files = []
total_subs = 0


def rewrite(text: str) -> tuple[str, int]:
    n = 0
    for old, new in MAP.items():
        # from old.xxx  /  from old import  /  from old(  EOL
        pat_from = re.compile(rf"(^|\n)(\s*from\s+){old}(\.|(\s+import))")
        text, c = pat_from.subn(lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(3)}", text)
        n += c
        # import old.xxx  /  import old as  /  import old(EOL or ,)
        pat_imp = re.compile(rf"(^|\n)(\s*import\s+){old}(\.|\s+as\s|\s*$|\s*#|,)", re.MULTILINE)
        text, c = pat_imp.subn(lambda m: f"{m.group(1)}{m.group(2)}{new}{m.group(3)}", text)
        n += c
    # model.* 旧命名空间(出现在字符串 monkeypatch / import_module / 真实 import 均按子串安全替换,
    # 因为这些是完整模块路径, 不会误伤 model.safetensors / model.pt(后缀不在 map 中))
    for old, new in MODEL_MODULE_MAP.items():
        if old in text:
            text = text.replace(old, new)
            n += 1
    return text, n


for d in TARGET_DIRS:
    for py in d.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        src = py.read_text(encoding="utf-8")
        new, n = rewrite(src)
        if n and new != src:
            py.write_text(new, encoding="utf-8")
            changed_files.append((str(py.relative_to(ROOT)), n))
            total_subs += n

print(f"Changed {len(changed_files)} files, {total_subs} substitution-passes")
for f, n in sorted(changed_files):
    print(f"  {n:3d}  {f}")
