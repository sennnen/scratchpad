#!/bin/bash
until grep -q P3_DONE /home/user/nets/p3.log; do sleep 30; done
D=/home/user/data/p2; O=/home/user/nets/p4; mkdir -p $O
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$D/VG.l5000.ft vt=$D/VT.l5000.ft --reinit --quiet --nkeep 186365 --lr 3e-3 --steps 1500"
for L in teacher search; do for A in G T; do
  [ -f $O/${L}_$A.nnue ] || $FT --target $L --train $D/$A.l5000.ft --out $O/${L}_$A.nnue > $O/${L}_$A.log 2>&1
  python3 -c "import json; j=json.load(open('$O/${L}_$A.nnue.json')); print('${L}_$A', {k:round(v['mse'],6) for k,v in j['val_after'].items()}, j['train_sizes'])"
done; done
echo P4_TRAIN_DONE
cd /home/user/scratchpad; R=experiments/p4/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$O/$1.nnue $2=$O/$2.nnue --nodes 20000 --rounds 500 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"; }
m teacher_T teacher_G
m search_T search_G
echo P4_DONE
