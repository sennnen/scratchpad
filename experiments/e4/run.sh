#!/bin/bash
# E4: capacity scaling. Same run1 data (7,490 generation CPU-s); students H in {512,1024}.
T=/home/user/nets/trainer; D=/home/user/data/run1; O=/home/user/nets/e4; mkdir -p $O
TR="--train $D/d1.bin --train $D/d2.bin --train $D/d3.bin --train $D/d4.bin"
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1 --val vstree=/home/user/data/val/vs.bin:1"
tr() { name=$1; shift; [ -f $O/$name.stud ] || $T $TR $VAL --out $O/$name.stud "$@" 2> $O/$name.log; grep -E "val|wall" $O/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name"; }
tr tree_h512 --src 1 --H 512 --positions 1.5e8
tr tree_h1024 --src 1 --H 1024 --positions 1.5e8
tr game_h512 --src 0 --H 512 --positions 1.5e7
tr game_h1024 --src 0 --H 1024 --positions 1.5e7
cp /home/user/nets/e1/tree_teacher.stud $O/tree_h256.stud; cp /home/user/nets/e1/game_teacher_p1.5e7.stud $O/game_h256.stud
cd /home/user/scratchpad; R=experiments/e4/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$O/$1.stud $2=$O/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m tree_h512 tree_h256; m tree_h1024 tree_h512; m game_h512 game_h256; m game_h1024 game_h512
echo E4_DONE
