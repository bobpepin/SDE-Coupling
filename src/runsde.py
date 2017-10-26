import numpy as np
import sde
from xdmf import Xdmf3File
from tic import tic

n = 1
h = 1e-2

x0 = 0
t0 = 0

TOTAL = 100_000
N = 100
S = TOTAL // N

def b(t, x):
    return -x

def btrunc(t, x):
    return np.max((-1/h, np.min((1/h, b(t, x)))))

sigma = np.sqrt(2)*np.matlib.eye(n)

X = np.zeros((S, N))
B = np.zeros((S, N))

with tic:
    for i in range(0, S):
        X1, B1, t = sde.sde(x0, t0, b, sigma, N, h)
        (X[i, :], B[i, :]) = (np.squeeze(X1), np.squeeze(B1))

print("Rate (N={}, S={}): {:.1e} samples/s".format(N, S, TOTAL / tic.delta))
        
with Xdmf3File('sde', 'w') as f:
    f.add_scalar('X', X)
    f.add_scalar('B', B)
