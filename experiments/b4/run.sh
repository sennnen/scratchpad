#!/bin/bash
# B4: does the tree-position advantage arise in feature learning or in the readout? Freeze the feature
# transformer of a game-trained (FTg = E1 game_teacher_p1.5e7) or tree-trained (FTt = E1 tree_teacher)
# H=256 student, re-initialise the output layer, train it on 734k game (G) or 734k tree (T) positions.
until grep -q B2_DONE /home/user/nets/b2.log 2>/dev/null; do sleep 30; done
T=/home/user/nets/trainer.p2e; N=/home/user/nets/b4; mkdir -p $N
R1=$(for i in 1 2 3 4; do echo --train /home/user/data/run1/d$i.bin; done)
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
FTg=/home/user/nets/e1/game_teacher_p1.5e7.stud; FTt=/home/user/nets/e1/tree_teacher.stud
trn() { name=$1; init=$2; src=$3; [ -f $N/$name.stud ] || $T $R1 $VAL --src $src --n 734092 --positions 1.5e7 --init $init --freeze_ft --reinit_out --out $N/$name.stud 2> $N/$name.log
  grep val $N/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name"; }
trn FTg_G $FTg 0; trn FTg_T $FTg 1; trn FTt_G $FTt 0; trn FTt_T $FTt 1
cd /home/user/scratchpad; R=experiments/b4/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m FTg_T FTg_G; m FTt_T FTt_G; m FTt_G FTg_G
echo B4_DONE
