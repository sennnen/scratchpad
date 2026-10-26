#!/bin/bash
# E1b: data scaling. frac = fraction of run1 games (= fraction of generation CPU).
T=/home/user/nets/trainer; D=/home/user/data/run1; O=/home/user/nets/e1b; mkdir -p $O
TR="--train $D/d1.bin --train $D/d2.bin --train $D/d3.bin --train $D/d4.bin"
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
tr() { name=$1; shift; [ -f $O/$name.stud ] || $T $TR $VAL --out $O/$name.stud "$@" 2> $O/$name.log; grep val $O/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name"; }
tr tree_f32 --src 1 --frac 0.03125 --positions 1.6e7
tr tree_f16 --src 1 --frac 0.0625 --positions 3.2e7
tr tree_f8 --src 1 --frac 0.125 --positions 6.4e7
tr tree_f4 --src 1 --frac 0.25 --positions 1e8
tr tree_f2 --src 1 --frac 0.5 --positions 1e8
tr game_f4 --src 0 --frac 0.25 --positions 3.7e6
tr game_f2 --src 0 --frac 0.5 --positions 7.4e6
cp /home/user/nets/e1/tree_teacher.stud $O/tree_f1.stud
cp /home/user/nets/e1/game_teacher_p1.5e7.stud $O/game_f1.stud
cd /home/user/scratchpad
m() { [ -f experiments/e1b/matches/$1_vs_$2.json ] || python3 tools/match.py $1=$O/$1.stud $2=$O/$2.stud --nodes 20000 --rounds 100 --out experiments/e1b/matches/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m game_f2 game_f1; m game_f4 game_f2
m tree_f32 game_f1; m tree_f16 tree_f32; m tree_f8 tree_f16; m tree_f4 tree_f8; m tree_f2 tree_f4; m tree_f1 tree_f2
m tree_f32 game_f2; m tree_f8 game_f1
echo E1B_DONE
