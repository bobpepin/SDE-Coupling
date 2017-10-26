import sys
import os
import threading
import pdb
import cffi
import numpy as np
import asdf
from tic import tic

def generate_experiment(indata):
    def val(x):
        if isinstance(x, dict):
            x = x.copy()
            start = x.pop('start')
            stop = x.pop('stop')
            spacing = x.pop('spacing', 'linspace')
            fn = getattr(np, spacing)
            return fn(start, stop, **x)
        elif isinstance(x, str):
            return x
        elif isinstance(x, list):
            return x
        else:
            return np.array(x, ndmin=1)
    exper = indata.tree
    outdata = asdf.AsdfFile()
    outdata.tree['experiment'] = exper
    # outdata.tree['omega_X'] = val(exper['omega_X'])
    # outdata.tree['omega_Y'] = val(exper['omega_Y'])
    params = {k: val(v) for (k, v) in exper['params'].items()}
    # params['invepsilon'] = [2**p for p in params['invepsilon_powers']]
    # del params['invepsilon_powers']
    if 'h' not in params:
      h = 0.1 / (np.max(params['kappa_X']) * np.max(params['invepsilon']))
      params['h'] = [h]

    variables = [k for k in sorted(params) if len(params[k]) > 1 and not isinstance(params[k], str)]
    variables += exper.get('extra_variables', [])
    suffix_format = ",".join(f"{k}={{{k}}}" for k in variables)
#    pdb.set_trace()
    def traverse(variables, path):
        if len(variables) == 0:
            yield path
        else:
            var = variables[0]
            for val in params[var]:
                yield from traverse(variables[1:], path + [(var, val)])
    
    g = traverse(variables, [])
    for i,sample in enumerate(g):
      p = { k: v[0] for k,v in params.items() if len(v) == 1 }
      p.update(sample)
      # pdb.set_trace()
      outdata.tree['params'] = p
      suffix = "_" + suffix_format.format(**dict(sample))
      prefix = indata.tree.get('prefix', '')
      outdata.tree['prefix'] = prefix
      outdata.tree['suffix'] = suffix
      outdata.tree['variables'] = variables
      outdata.tree['input_id'] = i
      fname = prefix + 'input' + suffix + '.asdf'
      os.makedirs(os.path.dirname(fname), exist_ok=True)
      outdata.write_to(fname)
      print(fname)

generate_experiment(asdf.AsdfFile.open(sys.argv[1]))
# print(list(v))
