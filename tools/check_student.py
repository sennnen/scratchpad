"""Check engine-side student eval (Rec.score on tree records recorded with a student loaded)
against an independent numpy forward pass of the same .stud file."""
import sys, struct, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from data import load, features

def load_stud(p):
    b = open(p, 'rb').read()
    assert b[:8] == b'STUDNT01'
    H, NB, sc = struct.unpack('<iif', b[8:20])
    a = np.frombuffer(b[20:], dtype='<f4')
    o = 0
    ftW = a[o:o + 768 * H].reshape(768, H); o += 768 * H
    ftB = a[o:o + H]; o += H
    outW = a[o:o + NB * 2 * H].reshape(NB, 2 * H); o += NB * 2 * H
    outB = a[o:o + NB]
    return H, NB, sc, ftW, ftB, outW, outB

H, NB, sc, ftW, ftB, outW, outB = load_stud(sys.argv[1])
r = load([sys.argv[2]], src=1)[:5000]
s, n, bk = features(r)
W = np.vstack([ftW, np.zeros((1, H), np.float32)])
acc_s = W[s.numpy()].sum(1) + ftB
acc_n = W[n.numpy()].sum(1) + ftB
x = np.concatenate([np.clip(acc_s, 0, 1) ** 2, np.clip(acc_n, 0, 1) ** 2], 1)
o = (x * outW[bk.numpy()]).sum(1) + outB[bk.numpy()]
pred = np.round(o * sc)
diff = np.abs(pred - r['score'])
print('n', len(r), 'max abs diff', diff.max(), 'mean', diff.mean(), 'teacher-student corr', np.corrcoef(r['teacher'].clip(-3000,3000), r['score'])[0,1])
