"""Write game-position records whose played move (== search bestmove) was not a capture.
Game records of one game are contiguous and in order; a capture is detected when the next record
of the same game has exactly one piece fewer. (Equivalent of nnue-pytorch 'smart fen skipping';
in-check positions are already absent.) The last position of each run is dropped (unknown move)."""
import sys, numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0])
from data import load, REC

def popcnt(recs):
    b = recs['board']
    return ((b & 15) != 0).sum(1) + ((b >> 4) != 0).sum(1)

out = sys.argv[1]
recs = load(sys.argv[2:], src=0)
pc = popcnt(recs)
nxt = np.roll(pc, -1)
diff = pc - nxt
quiet = diff == 0  # next position same piece count (non-capture) ; game boundary rarely gives 0
quiet[-1] = False
print('total', len(recs), 'quiet', quiet.sum(), 'frac', quiet.mean())
recs[quiet].tofile(out)
