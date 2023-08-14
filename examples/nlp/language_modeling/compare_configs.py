import sys
import os
import omegaconf
from pprint import pprint

d = []
for fname in os.listdir(sys.argv[1]):
    x = omegaconf.OmegaConf.load(os.path.join(sys.argv[1], fname))
    y = omegaconf.OmegaConf.load(os.path.join(sys.argv[2], fname))

    import pdb; pdb.set_trace()

    # LIST OF MISSING CONFIG FIELDS!
    if 'retro' in fname:
        x.trainer.benchmark=False
        x.model.hysteresis=2
    if 'bert' in fname:
        x.model.hysteresis=1
    if 'bart' in fname:
        x.model.hysteresis=2
    if 't5' in fname:
        x.model.hysteresis=2

    # LIST OF EXRENEOUS FIELDS!
    if not 'retro' in fname:
        del y.model._target_

    if x != y:
        d.append((fname, x,y))

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

print('Number of mismatched configs: ', len(d))
for pair in d:
    print(pair[0])
    f1 = omegaconf.OmegaConf.to_container(pair[1])
    f1 = flatten_dict(f1)
    f2 = omegaconf.OmegaConf.to_container(pair[2])
    f2 = flatten_dict(f2)

    key_differences = set(f1.keys()).symmetric_difference(set(f2.keys()))
    if key_differences:
        print('Mismatched keys:', key_differences)
    else:
        diff = {k:f1[k] == f2[k] for k in f1}
        diff = {k:(f1[k], f2[k]) for k in f1 if not diff[k]}
        print('Diff:', diff)
