"""
Legacy code analyzer — extracts structure without requiring full file review.
Run: python analyze_legacy.py <legacy_file.py>
Output: legacy_summary.txt
"""

import ast
import re
import sys
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FunctionInfo:
    name: str
    line: int
    end_line: int
    loc: int
    args: list[str]
    scope: str
    docstring: str | None


@dataclass
class ClassInfo:
    name: str
    line: int
    methods: list[str]


@dataclass
class CommentSection:
    """A prominent comment block used as an informal section header."""
    line: int
    text: str


@dataclass
class TopLevelAssignment:
    """A module-level variable assignment (excluding ALL_CAPS constants)."""
    line: int
    names: list[str]
    value_summary: str


@dataclass
class AnalysisResult:
    total_lines: int
    code_lines: int
    blank_lines: int
    comment_lines: int
    parse_error: str | None
    imports: list[str]
    constants: list[str]
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    globals_used: list[str]
    comment_sections: list[CommentSection]
    top_level_assignments: list[TopLevelAssignment]

    @property
    def function_count(self) -> int:
        return len(self.functions)

    @property
    def large_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if f.loc > 50]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _summarize_value(node: ast.expr) -> str:
    """Return a short human-readable description of an AST value node."""
    if isinstance(node, ast.Constant):
        return repr(node.value)[:40]
    if isinstance(node, ast.Call):
        func = ast.unparse(node.func) if hasattr(ast, "unparse") else "call"
        return f"{func}(...)"
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        kind = {ast.List: "list", ast.Tuple: "tuple", ast.Set: "set"}[type(node)]
        return f"{kind}[{len(node.elts)} items]"
    if isinstance(node, ast.Dict):
        return f"dict[{len(node.keys)} keys]"
    if hasattr(ast, "unparse"):
        return ast.unparse(node)[:60]
    return type(node).__name__


def _extract_comment_sections(lines: list[str]) -> list[CommentSection]:
    """
    Identify comment blocks likely used as section separators.
    Heuristic: comment lines with 5+ repeated characters (e.g. ###, ---) or
    comment lines immediately preceded and followed by a blank line.
    """
    sections = []
    separator_re = re.compile(r"^#\s*[-=*#]{4,}")

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue

        is_separator = bool(separator_re.match(stripped))
        surrounded_by_blank = (
            (i == 0 or lines[i - 1].strip() == "")
            and (i + 1 >= len(lines) or lines[i + 1].strip() == "")
        )

        if is_separator or surrounded_by_blank:
            text = stripped.lstrip("#").strip()
            if text:
                sections.append(CommentSection(line=i + 1, text=text))

    return sections


def analyze(filepath: str) -> AnalysisResult:
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        source = f.read()

    lines = source.splitlines()
    total_lines = len(lines)
    blank_lines = sum(1 for l in lines if l.strip() == "")
    comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    code_lines = total_lines - blank_lines - comment_lines

    try:
        tree = ast.parse(source)
        parse_error = None
    except SyntaxError as e:
        return AnalysisResult(
            total_lines=total_lines,
            code_lines=code_lines,
            blank_lines=blank_lines,
            comment_lines=comment_lines,
            parse_error=str(e),
            imports=[],
            constants=[],
            classes=[],
            functions=[],
            globals_used=[],
            comment_sections=_extract_comment_sections(lines),
            top_level_assignments=[],
        )

    # Attach parent references for scope detection
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            child.parent = node  # type: ignore[attr-defined]

    imports: list[str] = []
    constants: list[str] = []
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    globals_used: list[str] = []
    top_level_assignments: list[TopLevelAssignment] = []

    for node in ast.walk(tree):
        parent = getattr(node, "parent", None)

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)

        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append(f"{module}.{alias.name}")

        elif isinstance(node, ast.FunctionDef):
            scope = (
                "method" if isinstance(parent, ast.ClassDef)
                else "top-level" if isinstance(parent, ast.Module)
                else "nested"
            )
            functions.append(FunctionInfo(
                name=node.name,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                loc=(node.end_lineno or node.lineno) - node.lineno,
                args=[a.arg for a in node.args.args],
                scope=scope,
                docstring=(ast.get_docstring(node) or "")[:120] or None,
            ))

        elif isinstance(node, ast.ClassDef) and isinstance(parent, ast.Module):
            classes.append(ClassInfo(
                name=node.name,
                line=node.lineno,
                methods=[n.name for n in ast.walk(node) if isinstance(n, ast.FunctionDef)],
            ))

        elif isinstance(node, ast.Assign) and isinstance(parent, ast.Module):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not names:
                continue
            all_caps = all(n.isupper() for n in names)
            if all_caps:
                constants.extend(names)
            else:
                top_level_assignments.append(TopLevelAssignment(
                    line=node.lineno,
                    names=names,
                    value_summary=_summarize_value(node.value),
                ))

        elif isinstance(node, ast.Global):
            globals_used.extend(node.names)

    return AnalysisResult(
        total_lines=total_lines,
        code_lines=code_lines,
        blank_lines=blank_lines,
        comment_lines=comment_lines,
        parse_error=parse_error,
        imports=sorted(set(imports)),
        constants=constants,
        classes=classes,
        functions=functions,
        globals_used=sorted(set(globals_used)),
        comment_sections=_extract_comment_sections(lines),
        top_level_assignments=top_level_assignments,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render(data: AnalysisResult) -> str:
    out: list[str] = []
    sep = "-" * 60

    out.append("=== LEGACY FILE ANALYSIS ===\n")
    out.append(f"Total lines : {data.total_lines}")
    out.append(f"Code lines  : {data.code_lines}")
    out.append(f"Blank lines : {data.blank_lines}")
    out.append(f"Comments    : {data.comment_lines}")
    out.append(f"Functions   : {data.function_count}")
    out.append(f"Top-level assignments : {len(data.top_level_assignments)}")

    if data.parse_error:
        out.append(f"\n[!] PARSE ERROR: {data.parse_error}")

    out.append(f"\n{sep}")
    out.append("IMPORTS")
    out.append(sep)
    for imp in data.imports:
        out.append(f"  {imp}")

    if data.constants:
        out.append(f"\n{sep}")
        out.append("CONSTANTS (ALL_CAPS)")
        out.append(sep)
        for c in data.constants:
            out.append(f"  {c}")

    if data.classes:
        out.append(f"\n{sep}")
        out.append("CLASSES")
        out.append(sep)
        for cls in data.classes:
            out.append(f"  class {cls.name} (line {cls.line})")
            for m in cls.methods:
                out.append(f"    .{m}()")

    out.append(f"\n{sep}")
    out.append("FUNCTIONS (name | line | loc | scope | args)")
    out.append(sep)
    for fn in data.functions:
        scope_tag = "" if fn.scope == "top-level" else f" [{fn.scope}]"
        args = ", ".join(fn.args) if fn.args else ""
        out.append(f"  {fn.name}({args})  — line {fn.line}, {fn.loc} loc{scope_tag}")
        if fn.docstring:
            out.append(f"    \"{fn.docstring}\"")

    if data.large_functions:
        out.append(f"\n{sep}")
        out.append("LARGE FUNCTIONS (>50 loc) — refactor candidates")
        out.append(sep)
        for fn in data.large_functions:
            out.append(f"  {fn.name}  — line {fn.line}, {fn.loc} loc")

    if data.globals_used:
        out.append(f"\n{sep}")
        out.append("GLOBAL VARIABLES USED (code smell)")
        out.append(sep)
        for g in data.globals_used:
            out.append(f"  {g}")

    if data.comment_sections:
        out.append(f"\n{sep}")
        out.append("COMMENT SECTION HEADERS (informal separators)")
        out.append(sep)
        for s in data.comment_sections:
            out.append(f"  line {s.line:4d}  {s.text}")

    if data.top_level_assignments:
        out.append(f"\n{sep}")
        out.append("TOP-LEVEL ASSIGNMENTS (module-level data flow)")
        out.append(sep)
        for a in data.top_level_assignments:
            names = ", ".join(a.names)
            out.append(f"  line {a.line:4d}  {names} = {a.value_summary}")

    return "\n".join(out)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_legacy.py <legacy_file.py>")
        sys.exit(1)

    result = analyze(sys.argv[1])
    report = render(result)

    output_path = "legacy_summary.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n[✓] Report saved to {output_path}")
