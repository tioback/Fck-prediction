"""
Legacy code analyzer — extracts structure without requiring full file review.
Run: python analyze_legacy.py <legacy_file.py>
Output: legacy_summary.txt
"""

import ast
import sys
import json
from collections import defaultdict


def analyze(filepath: str) -> dict:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    lines = source.splitlines()
    total_lines = len(lines)

    try:
        tree = ast.parse(source)
        parse_error = None
    except SyntaxError as e:
        tree = None
        parse_error = str(e)

    imports = []
    functions = []
    classes = []
    globals_used = []
    constants = []

    if tree:
        for node in ast.walk(tree):
            # Imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")

            # Top-level functions
            elif isinstance(node, ast.FunctionDef) and isinstance(
                getattr(node, "parent", None), type(None)
            ):
                pass  # handled below with parent tracking

            # Top-level constants (ALL_CAPS assignments)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        constants.append(target.id)

        # Track parents for scoping
        for node in ast.walk(tree):
            for child in ast.iter_child_nodes(node):
                child.parent = node  # type: ignore

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                parent = getattr(node, "parent", None)
                is_top_level = isinstance(parent, ast.Module)
                is_method = isinstance(parent, ast.ClassDef)

                args = [a.arg for a in node.args.args]
                docstring = ast.get_docstring(node) or ""
                entry = {
                    "name": node.name,
                    "line": node.lineno,
                    "end_line": node.end_lineno,
                    "args": args,
                    "docstring": docstring[:120] if docstring else None,
                    "scope": "method" if is_method else ("top-level" if is_top_level else "nested"),
                    "loc": (node.end_lineno or node.lineno) - node.lineno,
                }
                functions.append(entry)

            elif isinstance(node, ast.ClassDef):
                parent = getattr(node, "parent", None)
                if isinstance(parent, ast.Module):
                    method_names = [
                        n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)
                    ]
                    classes.append({
                        "name": node.name,
                        "line": node.lineno,
                        "methods": method_names,
                    })

        # Detect global keyword usage (code smell indicator)
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                globals_used.extend(node.names)

    # Blank lines and comment lines
    blank_lines = sum(1 for l in lines if l.strip() == "")
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    code_lines = total_lines - blank_lines - comment_lines

    # Large functions (potential refactor targets)
    large_functions = [f for f in functions if f["loc"] > 50]

    return {
        "total_lines": total_lines,
        "code_lines": code_lines,
        "blank_lines": blank_lines,
        "comment_lines": comment_lines,
        "parse_error": parse_error,
        "imports": sorted(set(imports)),
        "constants": constants,
        "classes": classes,
        "functions": functions,
        "large_functions": large_functions,
        "globals_used": sorted(set(globals_used)),
        "function_count": len(functions),
    }


def render(data: dict) -> str:
    lines = []
    sep = "-" * 60

    lines.append("=== LEGACY FILE ANALYSIS ===\n")
    lines.append(f"Total lines : {data['total_lines']}")
    lines.append(f"Code lines  : {data['code_lines']}")
    lines.append(f"Blank lines : {data['blank_lines']}")
    lines.append(f"Comments    : {data['comment_lines']}")
    lines.append(f"Functions   : {data['function_count']}")

    if data["parse_error"]:
        lines.append(f"\n[!] PARSE ERROR: {data['parse_error']}")

    lines.append(f"\n{sep}")
    lines.append("IMPORTS")
    lines.append(sep)
    for imp in data["imports"]:
        lines.append(f"  {imp}")

    if data["constants"]:
        lines.append(f"\n{sep}")
        lines.append("CONSTANTS (ALL_CAPS)")
        lines.append(sep)
        for c in data["constants"]:
            lines.append(f"  {c}")

    if data["classes"]:
        lines.append(f"\n{sep}")
        lines.append("CLASSES")
        lines.append(sep)
        for cls in data["classes"]:
            lines.append(f"  class {cls['name']} (line {cls['line']})")
            for m in cls["methods"]:
                lines.append(f"    .{m}()")

    lines.append(f"\n{sep}")
    lines.append("FUNCTIONS (name | line | loc | args)")
    lines.append(sep)
    for fn in data["functions"]:
        scope_tag = "" if fn["scope"] == "top-level" else f" [{fn['scope']}]"
        args = ", ".join(fn["args"]) if fn["args"] else ""
        lines.append(f"  {fn['name']}({args})  — line {fn['line']}, {fn['loc']} loc{scope_tag}")
        if fn["docstring"]:
            lines.append(f"    \"{fn['docstring']}\"")

    if data["large_functions"]:
        lines.append(f"\n{sep}")
        lines.append("LARGE FUNCTIONS (>50 loc) — refactor candidates")
        lines.append(sep)
        for fn in data["large_functions"]:
            lines.append(f"  {fn['name']}  — line {fn['line']}, {fn['loc']} loc")

    if data["globals_used"]:
        lines.append(f"\n{sep}")
        lines.append("GLOBAL VARIABLES USED (code smell)")
        lines.append(sep)
        for g in data["globals_used"]:
            lines.append(f"  {g}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_legacy.py <legacy_file.py>")
        sys.exit(1)

    filepath = sys.argv[1]
    data = analyze(filepath)
    report = render(data)

    output_path = "legacy_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[✓] Report saved to {output_path}")
