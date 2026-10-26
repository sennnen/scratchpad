#!/bin/bash
# E12: tree data from fast self-play with a high recording rate, at the E3 budget (4 x 468 s = 1,870 CPU-s).
# Reference: E3 best (20000 nodes/move, 8%) and n5000_r08.
G=/home/user/data/sf_research_kb; B=/home/user/data/UHO_Lichess_4852_v1.epd
O=/home/user/data/e12; N=/home/user/nets/e12; mkdir -p $O $N
gen() { tag=$1; nodes=$2; rate=$3
  for i in 1 2 3 4; do
    [ -f $O/$tag.$i.done ] && continue
    { (echo "gendata nodes $nodes games 10000000 out $O/$tag.$i.bin book $B seed $((7000+i)) treerate $rate"; echo quit) | timeout 468 $G > /dev/null 2>&1; touch $O/$tag.$i.done; } &
  done; wait; }
gen n300_r25 300 0.25
gen n1000_r25 1000 0.25
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
T=/home/user/nets/trainer.kb2
for t in n300_r25 n1000_r25; do
  [ -f $N/$t.stud ] || $T $(for i in 1 2 3 4; do echo --train $O/$t.$i.bin; done) $VAL --src 1 --positions 1e8 --out $N/$t.stud 2> $N/$t.log
  grep val $N/$t.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $t $(grep -o '"n_train_unique": [0-9]*' $N/$t.stud.json)"
done
cp /home/user/nets/e3/n20000_r08.stud /home/user/nets/e3/n5000_r08.stud $N/
cd /home/user/scratchpad; R=experiments/e12/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m n300_r25 n20000_r08; m n1000_r25 n20000_r08
echo E12_DONE
