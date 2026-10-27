#!/bin/bash
# B4r: replication of B4 at H=512 with independently trained feature sources (E4 game_h512 / tree_h512).
until grep -q C1_DONE /home/user/nets/c1.log 2>/dev/null; do sleep 30; done
T=/home/user/nets/trainer.p2e; N=/home/user/nets/b4r; mkdir -p $N
R1=$(for i in 1 2 3 4; do echo --train /home/user/data/run1/d$i.bin; done)
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
FTg=/home/user/nets/e4/game_h512.stud; FTt=/home/user/nets/e4/tree_h512.stud
trn() { name=$1; init=$2; src=$3; [ -f $N/$name.stud ] || $T $R1 $VAL --H 512 --src $src --n 734092 --positions 1.5e7 --seed 1 --init $init --freeze_ft --reinit_out --out $N/$name.stud 2> $N/$name.log
  grep val $N/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name"; }
trn FTg_G $FTg 0; trn FTg_T $FTg 1; trn FTt_G $FTt 0; trn FTt_T $FTt 1
cd /home/user/scratchpad; R=experiments/b4r/matches; export SF_BIN=/home/user/data/sf_research_kb
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 150 --seed 7 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m FTg_T FTg_G; m FTt_T FTt_G; m FTt_G FTg_G
echo B4R_DONE
