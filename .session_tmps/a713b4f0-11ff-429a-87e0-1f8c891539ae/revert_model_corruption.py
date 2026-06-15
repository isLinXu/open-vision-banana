#!/usr/bin/env python3
"""Undo the over-eager `model.X -> sparklm.X` substring replacement.

The first rewrite pass wrongly replaced *runtime attribute accesses* like
``model.config.eos_token_id`` and ``language_model.config`` (where ``model`` /
``language_model`` are local variables) into ``sparklm.config...``.

Correct rule:
  * A real *module import/patch* of the old namespace appears only as:
        import_module("model.X")  /  __import__("model.X" ...)
        monkeypatch.setattr("model.X. ...")
    -> those SHOULD point to "sparklm.X" (correct, keep).
  * Everything else `sparklm.<mod>` that we introduced was a variable access and
    must be reverted to `model.<mod>` / `language_model.<mod>` etc.

Strategy: we only *introduced* corruption for these exact module names. Revert
any occurrence of ``sparklm.<mod>`` that is NOT part of:
  - a `from sparklm.` / `import sparklm.` statement
  - an `import_module("sparklm...")` / `__import__("sparklm...")` string
  - a `monkeypatch.setattr("sparklm...` string
  - a docstring/comment mention `sparklm.<mod>` (harmless, but we revert text in
    code only — to be safe we revert ALL and then fix the few legit string ones)

Simpler & safe: revert the bare-attribute corruptions by reversing the exact
token boundaries: replace ``sparklm.<mod>`` back to ``model.<mod>`` ONLY when the
preceding character is one of: identifier-char (e.g. `language_`), `(`, `=`,
space, `,`, `[`, but NOT when the line is an import / the token is preceded by
`from `/`import `.
"""
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
TARGET_DIRS = [ROOT / "src" / "sparklm", ROOT / "tests"]

MODS = ["model_sparklm", "moe_v4", "attention_v4", "kv_cache_quant",
        "kv_cache", "components", "config", "tokenizer"]

# Build a regex that matches `<prefixtoken>sparklm.<mod>` where prefixtoken is an
# identifier fragment that means this is really `xxx_model.<mod>` (e.g.
# `language_sparklm.config` should be `language_model.config`). Handle that first.
LANG_FIX = re.compile(r"language_sparklm\b")

changed = []
for d in TARGET_DIRS:
    for py in d.rglob("*.py"):
        if "__pycache__" in str(py):
            continue
        text = py.read_text(encoding="utf-8")
        orig = text
        n = 0

        # 1) language_sparklm -> language_model (always wrong)
        text, c = LANG_FIX.subn("language_model", text); n += c

        # 2) Revert bare variable-attribute corruptions line by line.
        out_lines = []
        for line in text.splitlines():
            stripped = line.lstrip()
            is_import = stripped.startswith(("from ", "import "))
            is_modpatch = ("import_module(" in line) or ("__import__(" in line) \
                or ("monkeypatch.setattr(" in line) or ('setattr("' in line)
            if is_import or is_modpatch:
                out_lines.append(line)
                continue
            new = line
            for mod in MODS:
                # `sparklm.<mod>` used as attribute on a variable named model/xxx_model.
                # We only introduced these; revert to `model.<mod>`.
                # Guard: don't touch real package refs `sparklm.<mod>` that the
                # ORIGINAL code already had as `from sparklm....` (handled above).
                pat = re.compile(rf"\bsparklm\.{mod}\b")
                new2 = pat.sub(f"model.{mod}", new)
                if new2 != new:
                    n += 1
                    new = new2
            out_lines.append(new)
        text = "\n".join(out_lines)
        if text != orig and not orig.endswith("\n"):
            pass
        elif orig.endswith("\n") and not text.endswith("\n"):
            text += "\n"

        if text != orig:
            py.write_text(text, encoding="utf-8")
            changed.append((str(py.relative_to(ROOT)), n))

print(f"Reverted in {len(changed)} files")
for f, n in sorted(changed):
    print(f"  {n:3d}  {f}")
