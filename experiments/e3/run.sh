#!/bin/bash
# E3: generation config at a fixed generation budget (~1870 CPU-s = 4 procs x ~468 s, i.e. the
# compute behind e1b tree_f4). Vary nodes/move and record rate; train H=256 on all tree data.
G=/home/user/data/sf_research; B=/home/user/data/UHO_Lichess_4852_v1.epd
O=/home/user/data/e3; N=/home/user/nets/e3; mkdir -p $O $N
gen() { tag=$1; nodes=$2; rate=$3; secs=$4
  for i in 1 2 3 4; do
    [ -f $O/$tag.$i.done ] && continue
    { (echo "gendata nodes $nodes games 100000 out $O/$tag.$i.bin book $B seed $((3000+i)) treerate $rate"; echo quit) | timeout $secs $G > /dev/null 2>&1; touch $O/$tag.$i.done; } &
  done; wait; }
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
trn() { tag=$1; P=$2; [ -f $N/$tag.stud ] || /home/user/nets/trainer $(for i in 1 2 3 4; do echo --train $O/$tag.$i.bin; done) $VAL --src 1 --positions $P --out $N/$tag.stud 2> $N/$tag.log; grep -E "val|wall" $N/$tag.log | awk '{printf "%s ", $0}'; echo "<- $tag"; }
gen n1000_r02 1000 0.02 468
gen n1000_r08 1000 0.08 468
gen n5000_r08 5000 0.08 468
gen n20000_r08 20000 0.08 468
for t in n1000_r02 n1000_r08 n5000_r08 n20000_r08; do trn $t 1e8; done
echo E3_TRAIN_DONE
