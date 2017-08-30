import sys
import glob
import re
import pdb
import numpy as np
import h5py

# files = glob.glob('I_XY_*.h5')
files = sys.argv[1:]
r = re.compile(r"I_XY_.*_(.*)\.h5")
stds = {}
means = {}
for f in files:
    print(f, file=sys.stderr)
    k = int(float(r.match(f).group(1)))
    with h5py.File(f, 'r') as h5f:
        I_X = h5f['I_X'][:]
    s = np.nanstd(I_X)
    m = np.nanmean(I_X)
#    pdb.set_trace()
    stds[k] = s
    means[k] = m

print("epsilon,std,mean")
for k in sorted(stds.keys()):
    eps = 2**-k
    s = stds[k]
    print("{},{},{},{}".format(-np.log(eps), -np.log(stds[k]), means[k], np.log(s)/np.log(eps)))

