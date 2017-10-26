import sys
import glob
import numpy as np
import asdf

def extract_Ef(datafiles):
    fsums = {}
    fcounts = {}
    for fname in datafiles:
        with asdf.AsdfFile.open(fname) as f:
            variables = f.tree['variables']
            params = f.tree['params']
            fsum = np.sum(f.tree['X'], axis=0).squeeze()
            fcount = params['omega_count']
        xy = (params['x0'], params['y0'])
        del params['x0']
        del params['y0']
        vk = (tuple(params.items()), tuple((v, params[v]) for v in variables if v not in ('x0', 'y0')))
        try:
            fsums[vk][xy] += fsum
            fcounts[vk][xy] += fcount
        except KeyError:
            fsums.setdefault(vk, {})[xy] = fsum
            fcounts.setdefault(vk, {})[xy] = fcount
    for vk in fsums:
        for xy in fsums[vk]:
            fsums[vk][xy] /= fcounts[vk][xy]
    return fsums

def make_grid(Ef):
    (x, y) = zip(*Ef.keys())
    xx, yy = np.meshgrid(np.unique(x), np.unique(y))
    f0 = next(iter(Ef.values()))
    Pt = np.empty((len(f0), *xx.shape))
    for i in range(0, xx.shape[0]):
        for j in range(0, xx.shape[1]):
            Pt[:, i, j] = Ef[(xx[i, j], yy[i, j])]
    return Pt, xx, yy

files = glob.glob(sys.argv[1] + '/XY_*')
Ef = extract_Ef(files)
grids = { k: make_grid(Ef[k]) for k in Ef }

with asdf.AsdfFile.open(files[0]) as metafile:
    tree = {'experiment': metafile.tree['experiment']}

suffix = tree['experiment'].get('suffix', '')
prefix = tree['experiment'].get('prefix', '')
for ((params, vv), (Pt, xx, yy)) in grids.items():
    params = dict(params)
    vv = dict(vv)
    tree['params'] = params
    tree['Pt'] = Pt[:, :, :, np.newaxis]
    tree['xcoords'] = xx[:, :, np.newaxis]
    tree['ycoords'] = yy[:, :, np.newaxis]
    # vv = { v: tree['params'][v] for v in variables }
    suffix_format = ",".join(f"{k}={{{k}}}" for k in vv)
    tree['prefix'] = prefix
    tree['suffix'] = suffix + "_" + suffix_format.format(**vv)
    outfile = asdf.AsdfFile(tree)
    fname = tree['prefix'] + 'Pt' + tree['suffix'] + '.asdf'
    print(fname)
    outfile.write_to(fname)


