import pdb
import os
import copy
import numpy as np
import asdf

def variable_suffix(variables, sample):
    suffix_format = ",".join(f"{k}={{{k}}}" for k in variables)
    return suffix_format.format(**dict(sample))

def inputs(exper):
    def val(x):
        if isinstance(x, dict) and 'start' in x:
            x = x.copy()
            start = x.pop('start')
            stop = x.pop('stop')
            spacing = x.pop('spacing', 'linspace')
            fn = getattr(np, spacing)
            return fn(start, stop, **x)
        elif isinstance(x, list):
            return x
        else:
            return [x]
    def isvariable(v):
        return len(v) > 1
    
    params = {k: val(v) for (k, v) in exper['params'].items()}
    variables = [k for k in sorted(params) if isvariable(params[k])]
    variables += exper.get('extra_variables', [])
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
      tree = {}
      tree['experiment'] = copy.deepcopy(exper)
      tree['params'] = copy.deepcopy(p)
      suffix = "_" + variable_suffix(variables, sample)
      prefix = exper.get('prefix', '')
      tree['prefix'] = prefix
      tree['suffix'] = suffix
      tree['variables'] = variables
      tree['input_id'] = i
      yield tree

def inputs_from_asdf(fname):
    with asdf.AsdfFile.open(fname) as f:
        return inputs(f.tree)

def output_to_asdf(output, input, label='output', outdir='.'):
    fname_prefix = input.get('prefix', '')
    fname_suffix = input.get('suffix', '')

    with asdf.AsdfFile() as f:
        f.tree.update(output)
        f.tree['input'] = copy.deepcopy(input)
        f.tree['experiment'] = copy.deepcopy(input['experiment'])
        fname = f'{fname_prefix}{label}{fname_suffix}.asdf'
        os.makedirs(outdir, exist_ok=True)
        # pdb.set_trace()
        f.write_to(outdir + "/" + fname)
        print(fname)
