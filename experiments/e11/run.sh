#!/bin/bash
# E11: (a) does the game pipeline keep improving at 30 nodes/move? (b) does capacity help with the
# best game data (100-node self-play)? (R3 was measured on 5000-node game data only.)
G=/home/user/data/sf_research_kb; B=/home/user/data/UHO_Lichess_4852_v1.epd
O=/home/user/data/e10; N=/home/user/nets/e10
for i in 1 2 3 4; do
  [ -f $O/n30.$i.done ] && continue
  { (echo "gendata nodes 30 games 10000000 out $O/n30.$i.bin book $B seed $((6000+i)) treerate 0"; echo quit) | timeout 468 $G > /dev/null 2>&1; touch $O/n30.$i.done; } &
done; wait
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
T=/home/user/nets/trainer.kb2
trn() { name=$1; tag=$2; shift 2; [ -f $N/$name.stud ] || $T $(for i in 1 2 3 4; do echo --train $O/$tag.$i.bin; done) $VAL --src 0 "$@" --out $N/$name.stud 2> $N/$name.log
  grep val $N/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name $(grep -o '"n_train_unique": [0-9]*' $N/$name.stud.json)"; }
trn game_n30_p3e7 n30 --positions 3e7
trn game_n100_h512_p3e7 n100 --positions 3e7 --H 512
cd /home/user/scratchpad; R=experiments/e11/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m game_n30_p3e7 game_n100_p3e7; m game_n100_h512_p3e7 game_n100_p3e7
echo E11_DONE
