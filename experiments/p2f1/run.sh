#!/bin/bash
# P2-F1 (pre-registered, see RESEARCH_LOG): fine-tune master layer stacks on 5000-node search labels.
until grep -q LABEL_DONE /home/user/nets/p2_label.log; do sleep 20; done
D=/home/user/data/p2; O=/home/user/nets/p2f1; mkdir -p $O
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$D/VG.l5000.ft vt=$D/VT.l5000.ft --quiet --steps 1500"
Q=$(python3 -c "
import sys; sys.path.insert(0,'/home/user/scratchpad/tools/p2'); from ftdata import load
g=load('$D/G.l5000.ft')['rec']; t=load('$D/T.l5000.ft')['rec']
q=lambda r: int((((r['src']&0x80)==0)&(abs(r['score'])<20000)).sum()); print(min(q(g),q(t)))")
echo "Q (quiet positions per arm) = $Q"
# 1) hyperparameter selection on the mixed arm only
for LR in 1e-4 3e-4 1e-3; do
  [ -f $O/M_lr$LR.nnue ] || $FT --train $D/G.l5000.ft:1 $D/T.l5000.ft:1 --nkeep $Q --lr $LR --out $O/M_lr$LR.nnue > $O/M_lr$LR.log 2>&1
done
BEST=$(python3 -c "
import json
r={lr: json.load(open('$O/M_lr%s.nnue.json'%lr))['val_after'] for lr in ['1e-4','3e-4','1e-3']}
s={lr:(v['vg']['mse']+v['vt']['mse'])/2 for lr,v in r.items()}; print(min(s,key=s.get))")
echo "selected lr $BEST"; cp $O/M_lr$BEST.nnue $O/FT_M.nnue
# 2) arms with frozen hyperparameters
[ -f $O/FT_G.nnue ] || $FT --train $D/G.l5000.ft --nkeep $Q --lr $BEST --out $O/FT_G.nnue > $O/FT_G.log 2>&1
[ -f $O/FT_T.nnue ] || $FT --train $D/T.l5000.ft --nkeep $Q --lr $BEST --out $O/FT_T.nnue > $O/FT_T.log 2>&1
for a in FT_G FT_T FT_M; do python3 -c "
import json; j=json.load(open('$O/$a.nnue.json')) if '$a'!='FT_M' else json.load(open('$O/M_lr$BEST.nnue.json'))
print('$a', 'before', {k:round(v['mse'],5) for k,v in j['val_before'].items()}, 'after', {k:round(v['mse'],5) for k,v in j['val_after'].items()})"; done
echo P2F1_TRAIN_DONE
# 3) matches (20k nodes, UHO_4060_v4, paired)
cd /home/user/scratchpad; R=experiments/p2f1/matches; export SF_BIN=/home/user/data/sf_p2
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$3 $2=$4 --nodes 20000 --rounds $5 --seed ${6:-1} --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"; }
m FT_T FT_G $O/FT_T.nnue $O/FT_G.nnue 500
m FT_T master $O/FT_T.nnue teacher 500
m FT_G master $O/FT_G.nnue teacher 500
m FT_M master $O/FT_M.nnue teacher 500
echo P2F1_DONE
