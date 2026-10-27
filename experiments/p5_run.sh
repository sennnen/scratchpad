#!/bin/bash
until grep -q P4_DONE /home/user/nets/p4.log; do sleep 30; done
B=/home/user/data/UHO_Lichess_4852_v1.epd; G=/home/user/data/sf_p2c; D=/home/user/data/p5; O=/home/user/nets/p5; mkdir -p $D $O
NET=/home/user/work/Stockfish/src/nn-c9e480e92425.nnue; S=/home/user/nets/p3/RS_T.nnue
gen() { tag=$1; ev=$2
  for i in 1 2 3 4; do
    [ -f $D/$tag.$i.done ] || { (echo "setoption name EvalFile value $ev"; echo "gendata nodes 5000 games 100 out $D/$tag.$i.bin book $B seed $((9100+i)) treerate 0.08"; echo quit) | $G 2>&1 | grep -a "gendata games" | tail -n1 > $D/$tag.$i.log; touch $D/$tag.$i.done; } &
  done; wait; cat $D/$tag.*.log; }
gen on $S
gen off $NET
python3 - <<'PY'
import sys, numpy as np; sys.path.insert(0, '/home/user/scratchpad/tools')
from data import load
rng = np.random.default_rng(5)
for tag in ('on', 'off'):
    t = load(['/home/user/data/p5/%s.%d.bin' % (tag, i) for i in range(1, 5)], src=1)
    t[rng.choice(len(t), 600000, replace=False)].tofile('/home/user/data/p5/%s.in.bin' % tag); print(tag, len(t))
PY
for tag in on off; do [ -f $D/$tag.ft ] || { (echo "ftdump in $D/$tag.in.bin out $D/$tag.ft"; echo quit) | $G | grep ftdump; } & done; wait
V=/home/user/data/p2; P3=/home/user/data/p3
FT="python3 /home/user/scratchpad/tools/p2/finetune.py --net $NET --val vg=$V/VG.l5000.ft vt=$V/VT.l5000.ft --target teacher --reinit --lr 3e-3 --steps 3000"
for tag in on off; do
  [ -f $O/RS_$tag.nnue ] || $FT --train $P3/T.ft:1 $D/$tag.ft:1 --out $O/RS_$tag.nnue > $O/RS_$tag.log 2>&1
  python3 -c "import json; j=json.load(open('$O/RS_$tag.nnue.json')); print('RS_$tag', {k:round(v['mse'],6) for k,v in j['val_after'].items()})"
done
cd /home/user/scratchpad; R=experiments/p5/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
[ -f $R/RS_on_vs_RS_off.json ] || python3 tools/match.py RS_on=$O/RS_on.nnue RS_off=$O/RS_off.nnue --nodes 20000 --rounds 500 --out $R/RS_on_vs_RS_off | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"
echo P5_DONE
