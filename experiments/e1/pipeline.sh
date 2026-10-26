#!/bin/bash
while pgrep -x sf_research >/dev/null; do sleep 10; done
mkdir -p /home/user/data/val && cd /home/user/data/val
if [ ! -f v.bin ]; then
for i in 1 2 3 4; do (echo "gendata nodes 5000 games 60 out /home/user/data/val/v$i.bin book /home/user/data/UHO_Lichess_4852_v1.epd seed $((900+i)) treerate 0.02"; echo quit) | /home/user/data/sf_research 2>&1 | grep -a "gendata games" | tail -n1 & done; wait
cat v1.bin v2.bin v3.bin v4.bin > v.bin
fi
python3 -c "
import sys; sys.path.insert(0,'/home/user/scratchpad/tools')
from data import load
print(len(load(['/home/user/data/run1/d%d.bin'%i for i in range(1,5)],src=0)))" > /home/user/nets/n_game_tmp
mkdir -p /home/user/nets/e1 && mv /home/user/nets/n_game_tmp /home/user/nets/e1/n_game
echo "n_game $(cat /home/user/nets/e1/n_game)"
/home/user/scratchpad/experiments/e1/run_train.sh
echo PIPELINE_DONE
