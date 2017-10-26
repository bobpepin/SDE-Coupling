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
    # h = 2**-19
    N = np.uint64((t1 - t0) / h)

    # x0 = np.zeros(1, dtype=np.float32);
    # y0 = np.zeros(1, dtype=np.float32);
    x0 = np.array(params['x0'], ndmin=1)
    y0 = np.array(params['y0'], ndmin=1)

    rng = np.random.RandomState()
    rng.seed(params['omega_Y_seed'])
    omega_Y = rng.randint(0, 2**64, params['omega_Y_count'], 'uint64')
    rng.seed(params['omega_X_seed'])
    omega_X = rng.randint(0, 2**64, params['omega_X_count'], 'uint64')
    # omega_Y = infile.tree['omega_Y']
    # omega_X = infile.tree['omega_X']
    #omega_Y = 0
    # omega_X = np.arange(0, 200)

    fname_prefix = infile.tree['prefix']
    fname_suffix = infile.tree['suffix']
    # fname_suffix = '{}_{}_{}_{}'.format(params['kappa_X'], params['kappa_Y'],
    #                                     omega_Y, params['invepsilon'])

    # X_fname = 'X_{}.bin'.format(fname_suffix)
    # Y_fname = 'Y_{}.bin'.format(fname_suffix)
    # X = np.memmap(X_fname, mode='w+',
    #               shape=(omega_X.size, N, x0.shape[0]), dtype=np.float32)
    # Y = np.memmap(X_fname, mode='w+',
    #               shape=(omega_X.size, N, y0.shape[0]), dtype=np.float32)

    X = np.zeros((omega_Y.size, omega_X.size, N, x0.shape[0]), dtype=np.float32)
    Y = np.zeros((omega_Y.size, omega_X.size, N, y0.shape[0]), dtype=np.float32)
    # t = np.arange(0, N, dtype=np.float32) * h
    # t = np.tile(t, (omega_X.size, 1))

    with tic:
        for (i, oY) in enumerate(omega_Y):
            for (j, oX) in enumerate(omega_X):
                Xi = X[i, j, :, :]
                Yi = Y[i, j, :, :]
                sde_eigen.sde(params, oX, oY, h, t0, x0, y0, Xi, Yi)

    with tic:
        I_X = np.trapz(np.exp(X), dx=h, axis=2)
        I_Y = np.trapz(np.exp(Y), dx=h, axis=2)

    # with xdmf3.Xdmf3File('XY_{}.xmf'.format(fname_suffix), 'w') as f:
    #     f.add_vector('X', X)
    #     f.add_vector('Y', Y)
    #     f.add_scalar('t', t)

    with asdf.AsdfFile(infile) as f:
        # f.tree['X'] = X
        # f.tree['Y'] = Y
        f.tree['I_X'] = I_X
        f.tree['I_Y'] = I_Y
        fname = f'{fname_prefix}I_XY{fname_suffix}.asdf'
        f.write_to(fname)
        print(fname)

    with asdf.AsdfFile(infile) as f:
        f.tree['X'] = X
        f.tree['Y'] = Y
        fname = f'{fname_prefix}XY{fname_suffix}.asdf'
        f.write_to(fname)
        print(fname)

        
    # with xdmf3.Xdmf3File('I_XY_{}.xmf'.format(fname_suffix), 'w') as f:
    #     f.add_scalar('I_X', I_X)
    #     f.add_scalar('I_Y', I_Y)


for fname in sys.argv[1:]:
    simulate(fname)
