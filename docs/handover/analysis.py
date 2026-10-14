"""Static analysis of the project source (AST) used to generate the handover document.

Everything in the API-reference chapters (signatures, types, docstrings, constants, raised
exceptions, file side effects, import graph, call graph) is extracted from the code itself,
so the document cannot drift from what is actually implemented.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXCLUDE_DIRS = {"outputs", "checkpoints", "__pycache__", ".pytest_cache", ".streamlit", ".git", "handover"}
# Calls that write to disk / external state (used to list side effects).
IO_CALLS = {"open", "write_text", "write_bytes", "mkdir", "save", "imwrite", "savefig", "dump", "replace",
            "unlink", "rmtree", "urlretrieve", "extractall", "VideoWriter", "write", "makedirs"}


@dataclass
class FunctionInfo:
    name: str
    qualname: str
    module: str
    lineno: int
    end_lineno: int
    signature: str
    params: list[tuple[str, str, str]]          # (name, annotation, default)
    returns: str
    docstring: str
    decorators: list[str]
    raises: list[str]
    calls: list[str]                            # resolved project-internal qualified names
    io: list[str]
    is_method: bool = False
    called_by: list[str] = field(default_factory=list)


@dataclass
class ClassInfo:
    name: str
    qualname: str
    module: str
    lineno: int
    end_lineno: int
    bases: list[str]
    docstring: str
    methods: list[FunctionInfo]
    attributes: list[tuple[str, str]]           # (name, first assignment source)
    class_vars: list[tuple[str, str]]


@dataclass
class ModuleInfo:
    module: str
    path: str                                   # relative POSIX path
    lines: int
    docstring: str
    internal_imports: list[str]
    external_imports: list[str]
    imported_by: list[str]
    constants: list[tuple[str, str]]
    classes: list[ClassInfo]
    functions: list[FunctionInfo]
    is_entry_point: bool


def _src(node: Optional[ast.AST]) -> str:
    return "" if node is None else ast.unparse(node)


def _short(text: str, n: int = 110) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 3] + "..."


def module_name(path: Path) -> str:
    rel = path.relative_to(PROJECT_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or "main"


def iter_source_files() -> list[Path]:
    out = []
    for p in sorted(PROJECT_ROOT.rglob("*.py")):
        rel = p.relative_to(PROJECT_ROOT)
        if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
            continue
        out.append(p)
    return out


class _Analyzer:
    def __init__(self, files: list[Path]) -> None:
        self.files = files
        self.project_modules = {module_name(p) for p in files}
        self.top_packages = {m.split(".")[0] for m in self.project_modules}

    def is_internal(self, mod: str) -> bool:
        return mod.split(".")[0] in self.top_packages

    def function(self, node: ast.FunctionDef, module: str, prefix: str, alias: dict[str, str], cls: Optional[str]) -> FunctionInfo:
        args = node.args
        params: list[tuple[str, str, str]] = []
        positional = args.posonlyargs + args.args
        defaults = [None] * (len(positional) - len(args.defaults)) + list(args.defaults)
        for a, d in zip(positional, defaults):
            params.append((a.arg, _src(a.annotation), _src(d)))
        if args.vararg:
            params.append(("*" + args.vararg.arg, _src(args.vararg.annotation), ""))
        for a, d in zip(args.kwonlyargs, args.kw_defaults):
            params.append((a.arg, _src(a.annotation), _src(d)))
        if args.kwarg:
            params.append(("**" + args.kwarg.arg, _src(args.kwarg.annotation), ""))
        sig_parts = []
        for name, ann, default in params:
            s = name + (f": {ann}" if ann else "")
            if default:
                s += f" = {default}"
            sig_parts.append(s)
        ret = _src(node.returns)
        signature = f"{node.name}({', '.join(sig_parts)})" + (f" -> {ret}" if ret else "")
        raises, calls, io = [], set(), set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise) and sub.exc is not None:
                exc = sub.exc
                if isinstance(exc, ast.Call):
                    msg = ""
                    if exc.args:
                        msg = _short(_src(exc.args[0]), 90)
                    raises.append(f"{_src(exc.func)}: {msg}" if msg else _src(exc.func))
                else:
                    raises.append(_src(exc))
            if isinstance(sub, ast.Call):
                f = sub.func
                target = None
                if isinstance(f, ast.Name):
                    if f.id in alias:
                        target = alias[f.id]
                    elif f.id in IO_CALLS:
                        io.add(f.id)
                elif isinstance(f, ast.Attribute):
                    if f.attr in IO_CALLS and not (isinstance(f.value, ast.Name) and f.value.id in ("self",)):
                        io.add(f.attr)
                    base = f.value
                    if isinstance(base, ast.Name):
                        if base.id == "self" and cls:
                            target = f"{module}.{cls}.{f.attr}"
                        elif base.id in alias:
                            target = f"{alias[base.id]}.{f.attr}"
                if target and self.is_internal(target):
                    calls.add(target)
        # Exclude the noisy "write" on generic objects unless clearly file-related.
        if "write" in io and not ({"open", "VideoWriter"} & io):
            io.discard("write")
        if "replace" in io and "save" not in io:
            io.discard("replace")
        return FunctionInfo(
            name=node.name, qualname=f"{prefix}{node.name}", module=module, lineno=node.lineno,
            end_lineno=node.end_lineno or node.lineno, signature=signature, params=params, returns=ret,
            docstring=ast.get_docstring(node) or "", decorators=[_src(d) for d in node.decorator_list],
            raises=raises, calls=sorted(calls), io=sorted(io), is_method=cls is not None)

    def analyze(self) -> dict[str, ModuleInfo]:
        modules: dict[str, ModuleInfo] = {}
        for path in self.files:
            mod = module_name(path)
            text = path.read_text(encoding="utf-8")
            tree = ast.parse(text)
            internal, external = set(), set()
            alias: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        (internal if self.is_internal(a.name) else external).add(a.name)
                        alias[a.asname or a.name.split(".")[0]] = a.name if a.asname else a.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if self.is_internal(node.module):
                        internal.add(node.module)
                    else:
                        external.add(node.module.split(".")[0] if node.module != "__future__" else "__future__")
                    for a in node.names:
                        alias[a.asname or a.name] = f"{node.module}.{a.name}"
            # local definitions are resolvable too
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    alias[node.name] = f"{mod}.{node.name}"
            constants, classes, functions = [], [], []
            for node in tree.body:
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                    for t in targets:
                        if isinstance(t, ast.Name) and (t.id.isupper() or t.id.startswith("_") and t.id[1:].isupper()):
                            constants.append((t.id, _short(_src(node.value), 160)))
                elif isinstance(node, ast.FunctionDef):
                    functions.append(self.function(node, mod, f"{mod}.", alias, None))
                elif isinstance(node, ast.ClassDef):
                    methods, attrs, cvars = [], [], []
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            fi = self.function(item, mod, f"{mod}.{node.name}.", alias, node.name)
                            methods.append(fi)
                            if item.name == "__init__":
                                seen = set()
                                for sub in ast.walk(item):
                                    if isinstance(sub, (ast.Assign, ast.AnnAssign)):
                                        tg = sub.targets if isinstance(sub, ast.Assign) else [sub.target]
                                        for t in tg:
                                            elts = t.elts if isinstance(t, ast.Tuple) else [t]
                                            for e in elts:
                                                if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) \
                                                        and e.value.id == "self" and e.attr not in seen:
                                                    seen.add(e.attr)
                                                    attrs.append((e.attr, _short(_src(sub.value), 100)))
                        elif isinstance(item, (ast.Assign, ast.AnnAssign)):
                            tg = item.targets if isinstance(item, ast.Assign) else [item.target]
                            for t in tg:
                                if isinstance(t, ast.Name):
                                    cvars.append((t.id, _short(_src(item.value) if item.value is not None else _src(item.annotation), 100)))
                    classes.append(ClassInfo(node.name, f"{mod}.{node.name}", mod, node.lineno, node.end_lineno or node.lineno,
                                             [_src(b) for b in node.bases], ast.get_docstring(node) or "", methods, attrs, cvars))
            is_entry = 'if __name__ == "__main__"' in text or "if __name__ == '__main__'" in text
            modules[mod] = ModuleInfo(mod, path.relative_to(PROJECT_ROOT).as_posix(), len(text.splitlines()),
                                      ast.get_docstring(tree) or "", sorted(internal), sorted(external - {"__future__"}),
                                      [], constants, classes, functions, is_entry)
        # Reverse imports.
        for m in modules.values():
            for imp in m.internal_imports:
                target = imp if imp in modules else None
                if target is None:  # "from a.b import c" where a.b.c is a module
                    continue
                modules[target].imported_by.append(m.module)
        # Resolve "from pkg import module" style imports for reverse index as well.
        for m in modules.values():
            m.imported_by = sorted(set(m.imported_by))
        # Reverse call graph.
        index: dict[str, FunctionInfo] = {}
        for m in modules.values():
            for f in m.functions:
                index[f.qualname] = f
            for c in m.classes:
                for f in c.methods:
                    index[f.qualname] = f
        class_names = {c.qualname for m in modules.values() for c in m.classes}
        for m in modules.values():
            for caller in [*m.functions, *[f for c in m.classes for f in c.methods]]:
                for callee in caller.calls:
                    key = callee
                    if key in class_names:  # constructing a class calls its __init__
                        key = f"{callee}.__init__"
                    if key in index and caller.qualname != key:
                        index[key].called_by.append(caller.qualname)
        for f in index.values():
            f.called_by = sorted(set(f.called_by))
        return modules


def analyze_project() -> dict[str, ModuleInfo]:
    return _Analyzer(iter_source_files()).analyze()


if __name__ == "__main__":
    mods = analyze_project()
    n_cls = sum(len(m.classes) for m in mods.values())
    n_fn = sum(len(m.functions) + sum(len(c.methods) for c in m.classes) for m in mods.values())
    missing = [f"{m.path}:{f.qualname.split('.', len(m.module.split('.')))[-1]}" for m in mods.values()
               for f in [*m.functions, *[x for c in m.classes for x in c.methods]] if not f.docstring]
    print(f"modules {len(mods)}, classes {n_cls}, functions/methods {n_fn}, without docstring {len(missing)}")
    for line in missing:
        print("  ", line)
