import sys
import cffi
import numpy as np
import threading
import xdmf3
from tic import tic

ffi = cffi.FFI()
ffi.cdef("""
typedef struct { float kappa_X, kappa_Y, invepsilon; } params_t;
""")
ffi.cdef("""
void sde_eigen(params_t *params,
               unsigned long omega_X, unsigned long omega_Y,
	       float h, unsigned long N, float t0,
	       unsigned long dimX, unsigned long dimY,
	       float *x0_p, float *y0_p,
	       float *X_p, float *Y_p);
""")

C = ffi.dlopen('sde_eigen.dylib')

def sde(params, omega_X, omega_Y, h, t0, x0, y0, X, Y):
    params_p = ffi.new("params_t *", params)
    x0_ptr = ffi.cast("float *", ffi.from_buffer(x0))
    y0_ptr = ffi.cast("float *", ffi.from_buffer(y0))
    X_ptr = ffi.cast("float *", ffi.from_buffer(X))
    Y_ptr = ffi.cast("float *", ffi.from_buffer(Y))
    dimX = x0.shape[0]
    dimY = y0.shape[0]
    N = X.shape[0]
    C.sde_eigen(params_p, omega_X, omega_Y, h, N, t0, dimX, dimY,
                x0_ptr, y0_ptr, X_ptr, Y_ptr)

invepsilon = np.float32(sys.argv[1])
params = { "invepsilon": invepsilon,
           "kappa_X": 1,
           "kappa_Y": 1 }

t0 = 0.0;
t1 = 1.0
h = 1e-5
N = np.uint64((t1 - t0) / h)

x0 = np.zeros(1, dtype=np.float32);
y0 = np.zeros(1, dtype=np.float32);

omega_Y = 0
omega_X = np.arange(0, 1000)

X = np.zeros((omega_X.size, N, x0.shape[0]), dtype=np.float32)
Y = np.zeros((omega_X.size, N, y0.shape[0]), dtype=np.float32)
t = np.arange(0, N, dtype=np.float32) * h
t = np.tile(t, (omega_X.size, 1))

with tic:
    for (i, omega) in enumerate(omega_X):
        Xi = X[i, :, :]
        Yi = Y[i, :, :]
        sde(params, omega, omega_Y, h, t0, x0, y0, Xi, Yi)

I_X = np.trapz(X, dx=h, axis=1)
I_Y = np.trapz(Y, dx=h, axis=1)
        
with xdmf3.Xdmf3File('XY_inveps={}.xmf'.format(params['epsilon']), 'w') as f:
    f.add_vector('X', X)
    f.add_vector('Y', Y)
    f.add_scalar('t', t)

with xdmf3.Xdmf3File('I_XY_inveps={}.xmf'.format(params['epsilon']), 'w') as f:
    f.add_scalar('I_X', I_X)
    f.add_scalar('I_Y', I_Y)

