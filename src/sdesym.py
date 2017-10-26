import sympy

N, M = sympy.symbols('N, M')
x = sympy.IndexedBase('X', shape=(N, M))
i, j = (sympy.Idx('i', 2), sympy.Idx('j', 2))
b = -x[i, j]
sympy.printing.print_ccode(b)

def grad(V):
    V = sympy.sympify(V)
    t = sympy.symbols('t')
    syms = sorted(V.free_symbols, key=sympy.Symbol.sort_key)
    X = sympy.MatrixSymbol('_X', len(syms), 1)
    subs = [(s, X[i, 0]) for (i, s) in enumerate(syms)]
    dVs = [sympy.diff(V, s).subs(subs) for s in syms]
    return sympy.lambdify((t, X), sympy.Matrix(dVs), 'numpy', dummify=False)
#    return lambdastr((t, X), sympy.Matrix(dVs), dummify=False)
