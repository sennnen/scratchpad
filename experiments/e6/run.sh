#!/bin/bash
# E6: without a teacher. Same label type (5000-node search score) and same number of labels:
# random tree positions (from run1 searches) vs quiet game positions. Isolates the position
# distribution for a conventional (search-labelled) pipeline.
D=/home/user/data/e6; N=/home/user/nets/e6; mkdir -p $D $N
python3 - <<'PY'
import sys; sys.path.insert(0, '/home/user/scratchpad/tools')
import numpy as np
from data import load
r = load(['/home/user/data/run1/d%d.bin' % i for i in range(1, 5)], src=1)
idx = np.random.default_rng(6).choice(len(r), 600000, replace=False)
s = r[np.sort(idx)]
for k in range(4): s[k::4].tofile('/home/user/data/e6/tree_in.%d.bin' % k)
PY
for k in 0 1 2 3; do
  [ -f $D/tree_lbl.$k.done ] || { (echo "label in $D/tree_in.$k.bin out $D/tree_lbl.$k.bin nodes 5000"; echo quit) | /home/user/data/sf_research_kb > $D/lbl.$k.log 2>&1; touch $D/tree_lbl.$k.done; } &
done; wait
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
T=/home/user/nets/trainer.kb
for P in 5e6 1.5e7; do
  [ -f $N/tree_score_p$P.stud ] || $T $(for k in 0 1 2 3; do echo --train $D/tree_lbl.$k.bin; done) $VAL --label score --positions $P --out $N/tree_score_p$P.stud 2> $N/tree_score_p$P.log
  grep val $N/tree_score_p$P.log | awk '{printf "%s %s ", $2, $4}'; echo "<- tree_score_p$P"
done
cp /home/user/nets/e1/gq_score_p1.5e7.stud $N/gq_score_p1.5e7.stud
cd /home/user/scratchpad; R=experiments/e6/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 150 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m tree_score_p1.5e7 gq_score_p1.5e7; m tree_score_p5e6 gq_score_p1.5e7
echo E6_DONE
