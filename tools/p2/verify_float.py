"""Verify the float/fake-quantized layer-stack model against Stockfish's integer eval on ftdump records."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from nnue_stacks import read_stacks, to_float
from ftdata import load

def act(h):  # pair activation with u8 quantization (floor), as Stockfish does
    sq = np.minimum(np.floor(128 * h * h), 127) / 128
    cl = np.clip(np.floor(128 * h), 0, 127) / 128
    return np.concatenate([sq, cl], -1)

def forward_np(st, ft, bucket, quant=True):
    out = np.zeros(len(ft))
    for b in np.unique(bucket):
        m = bucket == b; f = to_float(st[b])
        x = ft[m].astype(np.float64) / 128
        h0 = x @ f['W0'].T + f['b0']
        a0 = act(h0)
        h1 = a0 @ f['W1'].T + f['b1']
        a1 = act(h1)
        o = np.concatenate([a0, a1], -1) @ f['W2'].T + f['b2']
        out[m] = o[:, 0] + h0[:, 30] - h0[:, 31]
    return out

pre, st = read_stacks(sys.argv[1])
d = load(sys.argv[2])
o = forward_np(st, d['ft'], d['bucket'])
v = np.trunc(np.trunc(o * 16384) * 9600 / 16384) // 16  # approximate integer path
pred = o * 600
print('n', len(d), 'mean |600*o - sfEval|', np.abs(pred - d['sfeval']).mean(), 'max', np.abs(pred - d['sfeval']).max())
print('sfEval vs rec.teacher (tree recs) max diff', np.abs(d['sfeval'].astype(int) - d['rec']['teacher']).max())
