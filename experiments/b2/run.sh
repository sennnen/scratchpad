#!/bin/bash
# B2: error-guided selection of tree positions. Pool: B1 teacher-tree data. Scorer: E1 tree_teacher (H=256,
# trained on run1, disjoint games). Equal-count H=256 students (K=1M), weights err^alpha.
until grep -q P8_DONE /home/user/nets/p8.log 2>/dev/null; do sleep 30; done
O=/home/user/data/b1; N=/home/user/nets/b2; mkdir -p $N; S=/home/user/nets/e1/tree_teacher.stud
F=$(for i in 1 2 3 4; do echo --train $O/t.$i.bin; done)
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
T=/home/user/nets/trainer.p2b
trn() { name=$1; shift; [ -f $N/$name.stud ] || $T $F $VAL --src 1 --positions 3e7 "$@" --out $N/$name.stud 2> $N/$name.log
  grep -E "val|mined" $N/$name.log | awk '{printf "%s %s %s ", $1, $2, $4}'; echo "<- $name"; }
trn uniform --n 1000000
trn mine_a05 --mine $S --mine_k 1000000 --mine_alpha 0.5
trn mine_a1 --mine $S --mine_k 1000000 --mine_alpha 1
trn mine_a2 --mine $S --mine_k 1000000 --mine_alpha 2
cd /home/user/scratchpad; R=experiments/b2/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
for b in mine_a05 mine_a1 mine_a2; do m $b uniform; done
echo B2_DONE
