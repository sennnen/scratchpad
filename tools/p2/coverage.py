"""Mechanism analysis: redundancy and coverage of training position sets relative to the evaluator's
query distribution (held-out teacher search-tree positions)."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import load
import finetune as F

def feats(r):
    se, mat, r50 = F.board_stats(r)
    b = r['board']; codes = np.stack([b & 15, b >> 4], 2).reshape(len(r), 64)
    npc = (codes != 0).sum(1)
    imb = np.clip(np.round(se / 208), -9, 9)                      # material imbalance in pawns (stm view)
    pcb = np.clip(npc // 4, 0, 8)                                  # piece-count bucket
    ev = np.clip(np.round(np.sign(r['teacher']) * np.log1p(np.abs(r['teacher'].astype(float)) / 50)), -6, 6)
    return ((imb + 9) * 9 * 13 + pcb * 13 + (ev + 6)).astype(np.int64)  # joint cell id

def dist(r, ncell=19 * 9 * 13):
    return np.bincount(feats(r), minlength=ncell).astype(float) + 0.5

def kl(p, q):
    p = p / p.sum(); q = q / q.sum(); return float((p * np.log(p / q)).sum())

sets = {'G (5000-node games)': load(['/home/user/data/p3/G.in.bin']), 'F (100-node games)': load(['/home/user/data/p3/F.in.bin']),
        'T (tree)': load(['/home/user/data/p3/T.in.bin'])}
usage = load(['/home/user/data/val/v.bin'], src=1)  # held-out teacher-tree positions: the query distribution
pu = dist(usage)
for k, r in sets.items():
    uniq = len(np.unique(r['board'].view(np.dtype((np.void, 32))).ravel().tolist() if False else
                         np.ascontiguousarray(r['board']).view(np.dtype((np.void, 32)))))
    print(f"{k:22s} n={len(r)} unique-board frac {uniq/len(r):.3f}  KL(usage || set) {kl(pu, dist(r)):.3f}  "
          f"KL(set || usage) {kl(dist(r), pu):.3f}")

# Phase I: the query distribution of a *weak* student is its own search tree (vs.bin, student = E1 tree_teacher)
us = load(['/home/user/data/val/vs.bin'], src=1)
pus = dist(us)
print('vs weak-student query distribution (Phase I vs.bin):')
for k, r in sets.items():
    print(f"  {k:22s} KL(student-usage || set) {kl(pus, dist(r)):.3f}")
