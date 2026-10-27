"""Sample unlabelled position sets for Phase II (seeded). Train: run1 (master self-play, seeds 101-104);
validation: val/v.bin (seeds 901-904, disjoint games). Writes 4 shards per set for parallel labelling."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import load
rng = np.random.default_rng(20261026)
O = '/home/user/data/p2'
run1 = ['/home/user/data/run1/d%d.bin' % i for i in range(1, 5)]
def write(name, a):
    a = a[rng.permutation(len(a))]
    for k in range(4): a[k::4].tofile(f'{O}/{name}.in.{k}.bin')
    print(name, len(a))
g = load(run1, src=0); t = load(run1, src=1)
write('G', g[rng.choice(len(g), 300000, replace=False)])
write('T', t[rng.choice(len(t), 300000, replace=False)])
vg = load(['/home/user/data/val/v.bin'], src=0); vt = load(['/home/user/data/val/v.bin'], src=1)
write('VG', vg[rng.choice(len(vg), 20000, replace=False)])
write('VT', vt[rng.choice(len(vt), 20000, replace=False)])
