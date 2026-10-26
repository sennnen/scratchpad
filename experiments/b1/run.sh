#!/bin/bash
# B1 + B3 (mechanism): which tree positions carry the value? Equal-count (500k) H=256 students trained on
# teacher-tree positions from one boundary-annotated pool, restricted by ply-from-root band (B1) or by
# search context (B3: |scaled eval - beta|, qsearch vs main search), vs a uniform sample.
until grep -q P5_DONE /home/user/nets/p5.log 2>/dev/null; do sleep 30; done
G=/home/user/data/sf_p2c; B=/home/user/data/UHO_Lichess_4852_v1.epd; O=/home/user/data/b1; N=/home/user/nets/b1; mkdir -p $O $N
for i in 1 2 3 4; do
  [ -f $O/t.$i.done ] || { (echo "gendata nodes 5000 games 400 out $O/t.$i.bin book $B seed $((9000+i)) treerate 0.08 boundary 1"; echo quit) | $G 2>&1 | grep -a "gendata games" | tail -n1 > $O/t.$i.log; touch $O/t.$i.done; } &
done; wait; cat $O/t.*.log
F=$(for i in 1 2 3 4; do echo --train $O/t.$i.bin; done)
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
T=/home/user/nets/trainer.p2c; K=500000
trn() { name=$1; shift; [ -f $N/$name.stud ] || $T $F $VAL --src 1 --n $K --positions 1.5e7 "$@" --out $N/$name.stud 2> $N/$name.log
  grep val $N/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name $(grep -o '"n_train_unique": [0-9]*' $N/$name.stud.json)"; }
trn uniform
trn ply01_04 --plymin 1 --plymax 4
trn ply05_08 --plymin 5 --plymax 8
trn ply09_12 --plymin 9 --plymax 12
trn ply13_63 --plymin 13 --plymax 63
trn near100 --margin_lt 100
trn far600 --margin_gt 600
trn qsonly --qs 1
trn mainonly --qs 0
cd /home/user/scratchpad; R=experiments/b1/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds 100 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
for b in ply01_04 ply05_08 ply09_12 ply13_63 near100 far600 qsonly mainonly; do m $b uniform; done
echo B1_DONE
