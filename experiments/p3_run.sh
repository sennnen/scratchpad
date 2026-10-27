#!/bin/bash
until grep -q P2F1_DONE /home/user/nets/p2f1.log; do sleep 30; done
D=/home/user/data/p3; V=/home/user/data/p2; O=/home/user/nets/p3; mkdir -p $O
until [ $(grep -c "ftdump done" $D/*.dump.log | awk -F: '{s+=$2} END {print s}') -ge 3 ]; do sleep 20; done
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$V/VG.l5000.ft vt=$V/VT.l5000.ft --target teacher --reinit --steps 3000"
for LR in 3e-4 1e-3 3e-3; do
  [ -f $O/RSM_lr$LR.nnue ] || $FT --train $D/G.ft:1 $D/T.ft:1 --nkeep 300000 --lr $LR --out $O/RSM_lr$LR.nnue > $O/RSM_lr$LR.log 2>&1
done
BEST=$(python3 -c "
import json
r={lr: json.load(open('$O/RSM_lr%s.nnue.json'%lr))['val_after'] for lr in ['3e-4','1e-3','3e-3']}
s={lr:(v['vg']['mse']+v['vt']['mse'])/2 for lr,v in r.items()}; print(s, file=__import__('sys').stderr); print(min(s,key=s.get))")
echo "selected lr $BEST"
for A in G F T; do
  [ -f $O/RS_$A.nnue ] || $FT --train $D/$A.ft --lr $BEST --out $O/RS_$A.nnue > $O/RS_$A.log 2>&1
  python3 -c "import json; j=json.load(open('$O/RS_$A.nnue.json')); print('RS_$A', {k:round(v['mse'],6) for k,v in j['val_after'].items()})"
done
echo P3_TRAIN_DONE
cd /home/user/scratchpad; R=experiments/p3/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$3 $2=$4 --nodes 20000 --rounds $5 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"; }
m RS_T RS_F $O/RS_T.nnue $O/RS_F.nnue 500
m RS_T RS_G $O/RS_T.nnue $O/RS_G.nnue 300
m RS_T master $O/RS_T.nnue teacher 200
m RS_F master $O/RS_F.nnue teacher 200
echo P3_DONE
