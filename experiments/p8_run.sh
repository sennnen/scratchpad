#!/bin/bash
until grep -q B1_DONE /home/user/nets/b1.log; do sleep 30; done
D=/home/user/data/p8; V=/home/user/data/p2; O=/home/user/nets/p8; mkdir -p $D $O
for S in G T; do [ -f $D/$S.ft ] || { (echo "ftdump in /home/user/data/p3/$S.in.bin out $D/$S.ft"; echo quit) | /home/user/data/sf_p2d | grep ftdump; } & done; wait
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$V/VG.l5000.ft vt=$V/VT.l5000.ft --target teacher --reinit --lr 3e-3 --steps 3000 --see --nkeep 300000"
for S in T G; do
  [ -f $O/SEE_$S.nnue ] || $FT --train $D/$S.ft --out $O/SEE_$S.nnue > $O/SEE_$S.log 2>&1
  python3 -c "import json; j=json.load(open('$O/SEE_$S.nnue.json')); print('SEE_$S', {k:round(v['mse'],6) for k,v in j['val_after'].items()}, j['train_sizes'])"
done
cd /home/user/scratchpad; R=experiments/p8/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$3 $2=$4 --nodes 20000 --rounds $5 --seed 3 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"; }
m SEE_T RS_G_300000 $O/SEE_T.nnue /home/user/nets/p6/RS_G_300000.nnue 500
m SEE_G RS_G_300000 $O/SEE_G.nnue /home/user/nets/p6/RS_G_300000.nnue 300
echo P8_DONE
