import cffi
import numpy as np

ffi = cffi.FFI()
ffi.cdef("""
typedef struct { float kappa_X, kappa_Y, sigma_X, sigma_Y, invepsilon; } params_t;
""")
for d in ['ou', 'doublewell']:
    ffi.cdef("""
    void sde_eigen_{}(params_t *params,
               unsigned long omega_X, unsigned long omega_Y,
	       float h, unsigned long N, float t0,
	       unsigned long dimX, unsigned long dimY,
	       float *x0_p, float *y0_p,
	       float *X_p, float *Y_p);
    """.format(d))

C = ffi.dlopen('sde_eigen.dylib')

def sde(params, omega_X, omega_Y, h, t0, x0, y0, X, Y):
    x0 = np.ascontiguousarray(x0, dtype=np.float32)
    y0 = np.ascontiguousarray(y0, dtype=np.float32)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Y = np.ascontiguousarray(Y, dtype=np.float32)
    params_p = ffi.new("params_t *")
    for a in dir(params_p):
        setattr(params_p, a, params[a])
    x0_ptr = ffi.cast("float *", ffi.from_buffer(x0))
    y0_ptr = ffi.cast("float *", ffi.from_buffer(y0))
    X_ptr = ffi.cast("float *", ffi.from_buffer(X))
    Y_ptr = ffi.cast("float *", ffi.from_buffer(Y))
    dimX = x0.shape[0]
    dimY = y0.shape[0]
    N = X.shape[0]
    fn = getattr(C, 'sde_eigen_{}'.format(params['dynamics']))
    fn(params_p, omega_X, omega_Y, h, N, t0, dimX, dimY,
                x0_ptr, y0_ptr, X_ptr, Y_ptr)
