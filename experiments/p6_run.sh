#!/bin/bash
until grep -q P5_DONE /home/user/nets/p5.log; do sleep 30; done
D=/home/user/data/p3; V=/home/user/data/p2; O=/home/user/nets/p6; mkdir -p $O
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$V/VG.l5000.ft vt=$V/VT.l5000.ft --target teacher --reinit --lr 3e-3"
for n in 75000 150000 300000; do st=$([ $n -le 150000 ] && echo 1500 || echo 3000)
  for A in G T; do
    [ -f $O/RS_${A}_$n.nnue ] || $FT --train $D/$A.ft --nkeep $n --steps $st --out $O/RS_${A}_$n.nnue > $O/RS_${A}_$n.log 2>&1
    python3 -c "import json; j=json.load(open('$O/RS_${A}_$n.nnue.json')); print('RS_${A}_$n', {k:round(v['mse'],6) for k,v in j['val_after'].items()}, j['train_sizes'])"
  done; done
cp /home/user/nets/p3/RS_G.nnue $O/RS_G_600000.nnue; cp /home/user/nets/p3/RS_T.nnue $O/RS_T_600000.nnue
cd /home/user/scratchpad; R=experiments/p6/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
for n in 75000 150000 300000 600000; do for A in G T; do
  [ -f $R/RS_${A}_${n}_vs_master.json ] || python3 tools/match.py RS_${A}_$n=$O/RS_${A}_$n.nnue master=teacher --nodes 20000 --rounds 200 --out $R/RS_${A}_${n}_vs_master | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"
done; done
echo P6_DONE
