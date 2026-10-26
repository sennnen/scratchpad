#!/bin/bash
# E2: DAgger-style on-policy distillation.
# iter k: self-play with student S_{k-1} searching (nodes N), record a fraction of its evaluate()
# calls labelled by the teacher (master net), aggregate with previous data, retrain from scratch.
# Usage: dagger.sh START_STUD ITERS GAMES_PER_PROC TREERATE
set -e
S0=$1; ITERS=${2:-3}; G=${3:-300}; RATE=${4:-0.02}
T=/home/user/nets/trainer
O=/home/user/nets/e2; mkdir -p $O/data
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1 --val vstree=/home/user/data/val/vs.bin:1"
P=${P:-1e8}
prev=$S0
TRAIN="${BASE_TRAIN}"
for k in $(seq 1 $ITERS); do
  for i in 1 2 3 4; do
    f=$O/data/it${k}_$i.bin
    [ -f $f.done ] || { rm -f $f; (echo "setoption name StudentFile value $prev"; echo "gendata nodes 5000 games $G out $f book /home/user/data/UHO_Lichess_4852_v1.epd seed $((2000+100*k+i)) treerate $RATE"; echo quit) | /home/user/data/sf_research 2>&1 | grep -a "gendata games" | tail -n1 > $f.log && touch $f.done; } &
  done
  wait
  for i in 1 2 3 4; do TRAIN="$TRAIN --train $O/data/it${k}_$i.bin"; done
  name=dagger_it$k
  [ -f $O/$name.stud ] || $T $TRAIN $VAL --src 1 --label teacher --positions $P --out $O/$name.stud 2> $O/$name.log
  tail -n 4 $O/$name.log
  prev=$O/$name.stud
done
echo DAGGER_DONE
