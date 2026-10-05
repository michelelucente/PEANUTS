#!/usr/bin/env python3
"""
Structural check of the common-subexpression eliminations: every local name introduced only to
hold a repeated subexpression is substituted back with its defining expression, docstrings and
comments are dropped, and the resulting syntax tree must coincide with the reference one.

Usage: ast_equivalence.py <reference peanuts dir> <new peanuts dir> module:function[:names] ...
"""
import ast, copy, sys

class Sub(ast.NodeTransformer):
    def __init__(self, env): self.env = env
    def visit_Name(self, n):
        if isinstance(n.ctx, ast.Load) and n.id in self.env:
            return copy.deepcopy(self.env[n.id])
        return n

def inline(stmts, names, env):
    out = []
    for st in stmts:
        if (isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name)
                and st.targets[0].id in names):
            env[st.targets[0].id] = Sub(env).visit(st.value)
            continue
        if isinstance(st, ast.Expr) and isinstance(st.value, ast.Constant):
            continue  # docstring
        st = Sub(env).visit(st)
        for field in ('body', 'orelse'):
            if hasattr(st, field):
                setattr(st, field, inline(getattr(st, field), names, dict(env)))
        out.append(st)
    return out

def tree(path, func, names):
    t = ast.parse(open(path).read())
    fn = [n for n in t.body if isinstance(n, ast.FunctionDef) and n.name == func][0]
    fn.body = inline(fn.body, names, {})
    return ast.dump(fn, annotate_fields=False)

ok = True
ref, new = sys.argv[1], sys.argv[2]
for spec in sys.argv[3:]:
    mod, func, *rest = spec.split(':')
    names = set(rest[0].split(',')) if rest else set()
    same = tree(f"{ref}/{mod}.py", func, set()) == tree(f"{new}/{mod}.py", func, names)
    ok &= same
    print(f"{mod}.{func}: {'identical' if same else 'DIFFERENT'}")
sys.exit(0 if ok else 1)
