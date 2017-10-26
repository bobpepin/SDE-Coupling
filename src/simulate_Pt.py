import sys
import glob
import numpy as np
import asdf
from tic import tic
import sde_eigen

def simulate(fname):
    infile = asdf.AsdfFile.open(fname)

# power = np.float32(sys.argv[1])
# invepsilon = 2**power
# params = { "invepsilon": invepsilon,
#            "kappa_X": 1,
#            "kappa_Y": 1 }

    params = infile.tree['params']

    t0 = params['t0']
    t1 = params['t1']
    h = params['h']
    N = np.uint64((t1 - t0) / h)

    x0 = np.array(params['x0'], ndmin=1, dtype=np.float32)
    y0 = np.array(params['y0'], ndmin=1, dtype=np.float32)

    rng = np.random.RandomState()
    rng.seed(params['omega_seed'] + infile.tree['input_id'])
    count = params['omega_count']
    
    X = np.zeros((count, N, x0.shape[0]), dtype=np.float32)
    Y = np.zeros((count, N, y0.shape[0]), dtype=np.float32)

    for i in range(0, count):
        (oX, oY) = rng.randint(0, 2**64, 2, 'uint64')
        Xi = X[i, :, :]
        Yi = Y[i, :, :]
        sde_eigen.sde(params, oX, oY, h, t0, x0, y0, Xi, Yi)
            
    fname_prefix = infile.tree['prefix']
    fname_suffix = infile.tree['suffix']

    with asdf.AsdfFile(infile) as f:
        f.tree['X'] = X
        f.tree['Y'] = Y
        fname = f'{fname_prefix}XY{fname_suffix}.asdf'
        f.write_to(fname)
        print(fname)

with tic:
    files = glob.glob(sys.argv[1] + '/input_*')
    for fname in files:
        simulate(fname)
