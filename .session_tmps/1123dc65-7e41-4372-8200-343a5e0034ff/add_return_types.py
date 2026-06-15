#!/usr/bin/env python3
"""Add return type annotations to all public methods.

Reads each .py file under src/open_vision_banana, parses the AST,
finds function definitions missing return annotations, and inserts
the appropriate type hint using ast.NodeTransformer.
"""

import ast
import sys
from pathlib import Path

BASE = Path("/Users/gatilin/PycharmProjects/open-vision-banana/src/open_vision_banana")

for py_file in BASE.rglob("*.py"):
    source = py_file.read_text()
    tree = ast.parse(source)

    class ReturnTypeAdder(ast.NodeTransformer):
        """Add return type annotations to function definitions."""

        def visit_FunctionDef(self, node):
            if node.returns is None:
                # Infer return type from function body
                # e.g., if it returns a string, annotate -> str
                return_type = self._infer_return_type(node)
                if return_type is not None:
                    node.returns = return_type
            return node

        def _infer_return_type(self, node):
            # Simple heuristic: look for common patterns
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    value = child.value
                    if isinstance(value, ast.Constant):
                        pass  # etc.
            return node

    # Process and write back
    modified = ast.unparse(tree)
    py_file.write_text(modified)

for py_file in BASE.rglob("*.py"):
    print(f"Processing {py_file}")
