"""Check engine-side student eval (Rec.score on tree records recorded with a student loaded)
against an independent numpy forward pass of the same .stud file (STUDNT01 and STUDNT02/fset1)."""
import sys, struct, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from data import load

KB_OF = [0, 1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6] + [7] * 20

def load_stud(p):
    b = open(p, 'rb').read()
    v2 = b[:8] == b'STUDNT02'
    assert v2 or b[:8] == b'STUDNT01'
    H, NB, sc = struct.unpack('<iif', b[8:20]); o = 20; fset = 0
    if v2: fset = struct.unpack('<i', b[20:24])[0]; o = 24
    a = np.frombuffer(b[o:], dtype='<f4'); nf = 768 * (8 if fset == 1 else 1); o = 0
    ftW = a[o:o + nf * H].reshape(nf, H); o += nf * H
    ftB = a[o:o + H]; o += H
    outW = a[o:o + NB * 2 * H].reshape(NB, 2 * H); o += NB * 2 * H
    return H, NB, sc, fset, ftW, ftB, outW, a[o:o + NB]

def forward(net, rec):
    H, NB, sc, fset, ftW, ftB, outW, outB = net
    codes = []
    for sq in range(64):
        codes.append(int((rec['board'][sq >> 1] >> ((sq & 1) * 4)) & 15))
    kings = [codes.index(6), codes.index(14)]
    stm = int(rec['stm']); acc = []
    for c in (stm, 1 - stm):
        x = 56 * c; off = 0
        if fset == 1:
            ko = kings[c] ^ x; mir = 7 if (ko & 7) >= 4 else 0
            off = KB_OF[(ko >> 3) * 4 + ((ko & 7) ^ mir)] * 768; x ^= mir
        a = ftB.copy()
        for sq, code in enumerate(codes):
            if not code: continue
            rel = int((code >> 3) != c)
            a = a + ftW[off + (rel * 6 + (code & 7) - 1) * 64 + (sq ^ x)]
        acc.append(np.clip(a, 0, 1) ** 2)
    npc = sum(1 for cd in codes if cd)
    b = min(max((npc - 2) // 4, 0), NB - 1)
    return (np.concatenate(acc) * outW[b]).sum() + outB[b], sc

net = load_stud(sys.argv[1])
r = load([sys.argv[2]], src=1)[:int(sys.argv[3]) if len(sys.argv) > 3 else 2000]
d = []
for x in r:
    o, sc = forward(net, x)
    d.append(abs(round(o * sc) - int(x['score'])))
d = np.array(d)
print('fset', net[3], 'n', len(r), 'max abs diff', d.max(), 'mean', d.mean())
