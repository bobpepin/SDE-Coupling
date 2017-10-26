import numpy as np
import numpy.linalg as la
import numpy.matlib as mat

def sde(x0, t0, b, sigma, N, h):
    t = np.arange(t0, t0+N*h, h)
    #    t.shape = (t.shape[0], 1)
    #    N = t.size
    x0 = np.atleast_1d(x0)
    d = x0.shape[-1]
    dB = np.mat(np.random.normal(0, np.sqrt(h), (N, d)))
    dB = np.matlib.zeros((N, d))
    dB[0, :] = 0
    X = mat.zeros((N, d))
    X[:, 0] = x0
    for i in range(0, N-1):
        X[i+1, :] = X[i, :] + h*b(t[i], X[i, :]) + sigma*dB[i+1, :]
    B = np.cumsum(dB, axis=0)
    return (X, B, t)


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
