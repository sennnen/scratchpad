#!/bin/bash
# E1c: replication of "tree data at 1/32 compute beats game data at full compute" on disjoint
# 1/32 subsets of run1, and robustness of the comparison across node budgets.
T=/home/user/nets/trainer; D=/home/user/data/run1; O=/home/user/nets/e1c; mkdir -p $O
TR="--train $D/d1.bin --train $D/d2.bin --train $D/d3.bin --train $D/d4.bin"
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
for s in 0.25 0.5 0.75; do
  [ -f $O/tree_f32_s$s.stud ] || $T $TR $VAL --src 1 --frac 0.03125 --skip $s --positions 1.6e7 --out $O/tree_f32_s$s.stud 2> $O/tree_f32_s$s.log
  grep val $O/tree_f32_s$s.log | awk '{printf "%s %s ", $2, $4}'; echo "<- tree_f32_s$s"
done
cp /home/user/nets/e1/game_teacher_p1.5e7.stud $O/game_f1.stud; cp /home/user/nets/e1b/tree_f32.stud $O/tree_f32.stud
cp /home/user/nets/e1/tree_teacher.stud $O/tree_f1.stud
cd /home/user/scratchpad; R=experiments/e1c/matches
m() { [ -f $R/$1_vs_$2_n$3.json ] || python3 tools/match.py $1=$O/$1.stud $2=$O/$2.stud --nodes $3 --rounds 100 --out $R/$1_vs_$2_n$3 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
for s in 0.25 0.5 0.75; do m tree_f32_s$s game_f1 20000; done
m tree_f32 game_f1 5000; m tree_f32 game_f1 80000
m tree_f1 game_f1 80000
echo E1C_DONE
