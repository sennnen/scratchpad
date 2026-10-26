#!/bin/bash
# E13: headline ratio with the best settings on both sides. Tree: 20000 nodes/move, 8% (E3 best) at
# 234 and 468 CPU-s (4 procs x 59 / 117 s). Game: 100-node self-play, 1,870 CPU-s (E10 best).
G=/home/user/data/sf_research_kb; B=/home/user/data/UHO_Lichess_4852_v1.epd
O=/home/user/data/e13; N=/home/user/nets/e13; mkdir -p $O $N
gen() { tag=$1; secs=$2
  for i in 1 2 3 4; do
    [ -f $O/$tag.$i.done ] && continue
    { (echo "gendata nodes 20000 games 10000000 out $O/$tag.$i.bin book $B seed $((8000+i)) treerate 0.08"; echo quit) | timeout $secs $G > /dev/null 2>&1; touch $O/$tag.$i.done; } &
  done; wait; }
gen c234 59
gen c468 117
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
for t in c234 c468; do
  [ -f $N/tree_$t.stud ] || /home/user/nets/trainer.kb2 $(for i in 1 2 3 4; do echo --train $O/$t.$i.bin; done) $VAL --src 1 --positions 3.2e7 --out $N/tree_$t.stud 2> $N/tree_$t.log
  grep val $N/tree_$t.log | awk '{printf "%s %s ", $2, $4}'; echo "<- tree_$t $(grep -o '"n_train_unique": [0-9]*' $N/tree_$t.stud.json)"
done
cp /home/user/nets/e10/game_n100_p3e7.stud $N/
cd /home/user/scratchpad; R=experiments/e13/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 150 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m tree_c234 game_n100_p3e7; m tree_c468 game_n100_p3e7
echo E13_DONE
