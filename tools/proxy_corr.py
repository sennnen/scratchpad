"""Correlate validation MSE (game-position vs tree-position distributions) with fitted Elo.
Only teacher-label students (same target) are included so that MSE is comparable."""
import json, re, sys, glob, os
import numpy as np

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]

elo = {}
for line in open(sys.argv[1]):
    m = re.match(r"(\S+)\s+(-?[\d.]+)\s+\[", line)
    if m: elo[m.group(1)] = float(m.group(2))
alias = {"game_f1": "game_teacher_p1.5e7", "tree_f1": "tree_teacher"}
rows = []
for d in sys.argv[2:]:
    for f in glob.glob(d + "/*.stud.json"):
        j = json.load(open(f)); n = alias.get(os.path.basename(f)[:-10], os.path.basename(f)[:-10])
        if j["label"] != "teacher" or n not in elo or "vgame" not in j["val"]: continue
        rows.append((n, elo[n], j["val"]["vgame"]["mse_sig"], j["val"]["vtree"]["mse_sig"]))
rows = sorted(set(rows), key=lambda r: -r[1])
for r in rows: print(f"{r[0]:24s} elo {r[1]:7.1f} vgame {r[2]:.5f} vtree {r[3]:.5f}")
e = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows]); t = np.array([r[3] for r in rows])
print(f"n={len(rows)} spearman(elo,-vgame)={spearman(e,-g):.3f} spearman(elo,-vtree)={spearman(e,-t):.3f}")
print(f"pearson(elo,-vgame)={np.corrcoef(e,-g)[0,1]:.3f} pearson(elo,-vtree)={np.corrcoef(e,-t)[0,1]:.3f}")
