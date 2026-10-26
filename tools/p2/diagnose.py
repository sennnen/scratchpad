"""Master static eval (raw, mapped through scale_evaluation) vs N-node search score, by position source
and category. MSE in sigmoid space (scale 340)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from ftdata import load
import finetune as F
sig = lambda x: 1 / (1 + np.exp(-np.clip(x, -20000, 20000) / 340))
def stats(name, d):
    r = d['rec']; keep = np.abs(r['score']) < 20000; r = r[keep]; ev = d['sfeval'][keep].astype(float)
    se, mat, r50 = F.board_stats(r)
    scaled = F.scale_eval(ev, se, mat, r50)
    S = F.cp_to_internal(r['score'], r)
    err = (sig(scaled) - sig(S)) ** 2
    cap = (r['src'] & 0x80) > 0
    imb = np.abs(se) >= 300
    print(f"{name}: n={len(r)} capture-best {cap.mean():.2f} | MSE all {err.mean():.5f} quiet {err[~cap].mean():.5f} "
          f"capture {err[cap].mean():.5f} | quiet&imbalanced {err[~cap & imb].mean():.5f} (frac {np.mean(~cap & imb):.2f}) "
          f"quiet&balanced {err[~cap & ~imb].mean():.5f} | median|search| {np.median(np.abs(S)):.0f}")
    big = np.abs(scaled - S) > 300
    print(f"   share with |static-search| > 300 iu: all {big.mean():.3f} quiet {big[~cap].mean():.3f}")
for name, p in [('VG (game)', '/home/user/data/p2/VG.l5000.ft'), ('VT (tree)', '/home/user/data/p2/VT.l5000.ft')]:
    stats(name, load(p))
