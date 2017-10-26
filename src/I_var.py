import sys
import glob
import re
import pdb
import numpy as np
import asdf

# files = glob.glob('I_XY_*.h5')
files = sys.argv[1:]
# r = re.compile(r"I_XY_.*_(.*)\.h5")
stds = {}
means = {}
for fname in files:
    print(fname, file=sys.stderr)
    with asdf.AsdfFile.open(fname) as f:
        I_X = f.tree['I_X']
        variables = f.tree['variables']
        params = f.tree['params']
        k = tuple(params[v] for v in variables)
        s = np.std(I_X)
        m = np.mean(I_X)
        stds[k] = s
        means[k] = m

print(",".join(variables) + ",std,mean")
for k in sorted(stds.keys()):
    s = stds[k]
    m = means[k]
    print(",".join(repr(v) for v in (k + (s, m))))

