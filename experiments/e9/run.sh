#!/bin/bash
# E9: cheaper game-position baseline. 1000-node self-play (E3 n1000_r02 run, 1,870 CPU-s) gives ~750k game
# positions, 4x cheaper per position than run1's 5000-node games. Does it change R1's compute ratio?
T=/home/user/nets/trainer.kb2; O=/home/user/nets/e9; mkdir -p $O
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
F=$(for i in 1 2 3 4; do echo --train /home/user/data/e3/n1000_r02.$i.bin; done)
for P in 1e7 1.5e7; do
  [ -f $O/game_n1000_p$P.stud ] || $T $F $VAL --src 0 --positions $P --out $O/game_n1000_p$P.stud 2> $O/game_n1000_p$P.log
  grep val $O/game_n1000_p$P.log | awk '{printf "%s %s ", $2, $4}'; echo "<- game_n1000_p$P"
done
cp /home/user/nets/e1/game_teacher_p1.5e7.stud $O/game_f1.stud; cp /home/user/nets/e1b/tree_f32.stud $O/tree_f32.stud
cd /home/user/scratchpad; R=experiments/e9/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$O/$1.stud $2=$O/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m game_n1000_p1.5e7 game_f1; m tree_f32 game_n1000_p1.5e7
cp /home/user/nets/e8/e8_h1024_all.stud $O/e8_h1024_all.stud; cp /home/user/nets/e4/tree_h1024.stud $O/tree_h1024.stud
SF_BIN=/home/user/data/sf_research_kb m e8_h1024_all tree_h1024
echo E9_DONE
