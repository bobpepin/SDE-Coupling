import cffi
import numpy as np
import threading
from tic import tic

ffi = cffi.FFI()
ffi.cdef("""void sde(float *x0, float t0, float h,
            unsigned int N, unsigned int S, float *X);""")
ffi.cdef("""void sde_vec(float *x0, float t0, float h,
            unsigned int N, unsigned int S, float *X);""")
C = ffi.dlopen('sde_c.dylib')

x0 = np.zeros(1, dtype=np.float32);
t0 = 0.0;
h = 1e-2;

TOTAL = 2**31
N = 2**10
S = TOTAL // N

X = np.zeros((S, N), dtype=np.float32)
B = np.zeros((S, N), dtype=np.float32)

x0_ptr = ffi.cast("float *", ffi.from_buffer(x0))
X_ptr = ffi.cast("float *", ffi.from_buffer(X))

with tic:
    C.sde(x0_ptr, t0, h, N, S, X_ptr)
print("Rate (N={}, S={}): {:.1e} samples/s".format(N, S, TOTAL / tic.delta))

with tic:
    C.sde_vec(x0_ptr, t0, h, N, S, X_ptr)
print("Rate Vec (N={}, S={}): {:.1e} samples/s".format(N, S, TOTAL / tic.delta))

X1 = np.zeros((S//2, N), dtype=np.float32)
X2 = np.zeros((S//2, N), dtype=np.float32)
B = np.zeros((S, N), dtype=np.float32)

x0_ptr = ffi.cast("float *", ffi.from_buffer(x0))
X_ptr_1 = ffi.cast("float *", ffi.from_buffer(X1))
X_ptr_2 = ffi.cast("float *", ffi.from_buffer(X2))

th1 = threading.Thread(target=lambda: C.sde_vec(x0_ptr, t0, h, N, S//2, X_ptr_1))
th2 = threading.Thread(target=lambda: C.sde_vec(x0_ptr, t0, h, N, S//2, X_ptr_2))

with tic:
    th1.start()
    th2.start()
    th1.join()
    th2.join()
print("Rate Vec MT (N={}, S={}): {:.1e} samples/s".format(N, S, TOTAL / tic.delta))
