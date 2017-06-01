import numpy as np
import numpy.linalg as la
import numpy.matlib as mat
import sympy

def proj(sigma, z):
#    return 0
    nz = la.norm(z)
    if nz == 0:
        return mat.zeros(sigma.shape)
    sigmainv = la.inv(sigma)
    v = sigmainv*(z)
    n = la.norm(v)
    e = v / n
    return e*e.T

def sde_coupled(x0, y0, t0, b, sigma, N, h, delta=None):
    if delta is None:
        delta = h
#    t = np.arange(T[0], T[1]+h/2, h)
    t = np.arange(t0, t0+N*h, h)
    t.shape = (t.shape[0], 1)
#    N = t.size
    d = x0.shape[0]
    dB = np.mat(np.random.normal(0, np.sqrt(h), (d, N)))
    dB[:, 0] = 0
    X = mat.zeros((d, N))
    X[:, 0] = x0
    Y = mat.zeros((d, N))
    Y[:, 0] = y0
    Id = mat.eye(d)
    dBhat = mat.zeros((d, N))
    for i in range(0, N-1):
        dBhat[:, i+1] = (Id - 2*proj(sigma, X[:, i] - Y[:, i])) * dB[:, i+1]
        X[:, i+1] = X[:, i] + h*b(t[i], X[:, i]) + sigma*dB[:, i+1]
        Y[:, i+1] = Y[:, i] + h*b(t[i], Y[:, i]) + sigma*dBhat[:, i+1]
        if la.norm(X[:, i+1] - Y[:, i+1]) < delta:
            Y[:, i+1] = X[:, i+1]
        if d == 1 and (X[:, i] - Y[:, i]) * (X[:, i+1] - Y[:, i+1]) < 0:
            Y[:, i+1] = X[:, i+1]
    B = np.cumsum(dB, axis=1)
    Bhat = np.cumsum(dBhat, axis=1)
    return (X, Y, B, Bhat, t.T)

def grad(V):
    V = sympy.sympify(V)
    t = sympy.symbols('t')
    syms = sorted(V.free_symbols, key=sympy.Symbol.sort_key)
    X = sympy.MatrixSymbol('_X', len(syms), 1)
    subs = [(s, X[i, 0]) for (i, s) in enumerate(syms)]
    dVs = [sympy.diff(V, s).subs(subs) for s in syms]
    return sympy.lambdify((t, X), sympy.Matrix(dVs), 'numpy', dummify=False)
#    return lambdastr((t, X), sympy.Matrix(dVs), dummify=False)
    
def ou_coupled(x0, y0, kappa, sigma, h, dim):
    dB = np.mat(np.random.normal(0, h, dim))
    dB[:, 0] = 0
    X = mat.zeros(dim)
    X[:, 0] = x0
    Y = mat.zeros(dim)
    Y[:, 0] = y0
    Id = mat.eye(dim[0])
    dBhat = mat.zeros(dim)
    for i in range(0, dim[1]-1):
        dBhat[:, i+1] = (Id - 2*proj(sigma, X[:, i] - Y[:, i])) * dB[:, i+1]
        X[:, i+1] = X[:, i] - h*kappa*X[:, i] + sigma * dB[:, i+1]
        Y[:, i+1] = Y[:, i] - h*kappa*Y[:, i] + sigma * dBhat[:, i+1]
    B = np.cumsum(dB, axis=1)
    Bhat = np.cumsum(dBhat, axis=1)
    return (X, Y, B, Bhat)
