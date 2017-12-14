import os
import cffi
import numpy as np

dynamics = [('ou', 2), ('ou_r', 2), ('gd_r', 6), ('gd_r_3_2', 10)]

ffi = cffi.FFI()

hpath = os.path.dirname(os.path.realpath(__file__))
with open(hpath + '/sde.h') as hfile:
    ffi.cdef(hfile.read())
for (d, dim) in dynamics:
    ffi.cdef(f"""void sde_{d}_{dim}(
                   struct params_{d} params,
                   struct sde_input input,
                   struct sde_output output);""")

C = ffi.dlopen('sde.dylib')

def sde(dyn, params, h, x0, t0, omega, X):
    dim = X.shape[1]
    x0 = np.ascontiguousarray(x0, dtype=np.float32)
    x0_ptr = ffi.cast("float *", ffi.from_buffer(x0))
    omega = np.ascontiguousarray(omega, dtype=np.uint64)
    omega_ptr = ffi.cast("unsigned long *", ffi.from_buffer(omega))
    inp = { 'N': X.shape[0],
            'h': h,
            'x0': x0_ptr,
            'omega': omega_ptr,
            't0': t0 }
    X_ptr = ffi.cast("float *", ffi.from_buffer(X))
    fn = getattr(C, f'sde_{dyn}_{dim}')
    fn(params, inp, { 'X': X_ptr })
