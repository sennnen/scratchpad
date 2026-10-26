#!/bin/bash
# E10: is there a cheaper game-position pipeline that closes the gap? Game positions from very fast
# self-play (300 and 100 nodes/move) at the same 1,870 CPU-s budget as E3/E9 (4 procs x 468 s).
G=/home/user/data/sf_research_kb; B=/home/user/data/UHO_Lichess_4852_v1.epd
O=/home/user/data/e10; N=/home/user/nets/e10; mkdir -p $O $N
gen() { tag=$1; nodes=$2
  for i in 1 2 3 4; do
    [ -f $O/$tag.$i.done ] && continue
    { (echo "gendata nodes $nodes games 1000000 out $O/$tag.$i.bin book $B seed $((5000+i)) treerate 0"; echo quit) | timeout 468 $G > /dev/null 2>&1; touch $O/$tag.$i.done; } &
  done; wait; }
gen n300 300
gen n100 100
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
for t in n300 n100; do
  for P in 1.5e7 3e7; do
    [ -f $N/game_${t}_p$P.stud ] || /home/user/nets/trainer.kb2 $(for i in 1 2 3 4; do echo --train $O/$t.$i.bin; done) $VAL --src 0 --positions $P --out $N/game_${t}_p$P.stud 2> $N/game_${t}_p$P.log
    grep -E "val|n_train" $N/game_${t}_p$P.log | awk '{printf "%s %s ", $2, $4}'; echo "<- game_${t}_p$P $(grep -o '"n_train_unique": [0-9]*' $N/game_${t}_p$P.stud.json)"
  done
done
cp /home/user/nets/e9/game_n1000_p1.5e7.stud $N/; cp /home/user/nets/e1b/tree_f32.stud $N/
cd /home/user/scratchpad; R=experiments/e10/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m game_n300_p1.5e7 game_n1000_p1.5e7; m game_n300_p3e7 game_n1000_p1.5e7; m game_n100_p1.5e7 game_n1000_p1.5e7
echo E10_MATCHES1_DONE
