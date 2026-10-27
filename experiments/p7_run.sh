#!/bin/bash
until grep -q P6_DONE /home/user/nets/p6.log; do sleep 30; done
cd /home/user/scratchpad; R=experiments/p7/matches; mkdir -p $R; export SF_BIN=/home/user/data/sf_p2
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$3 $2=$4 --nodes 20000 --rounds $5 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*nnue//g"; }
m qT186 master /home/user/nets/p4/teacher_T.nnue teacher 200
m qG186 master /home/user/nets/p4/teacher_G.nnue teacher 200
m RS_T_300000 RS_G_300000 /home/user/nets/p6/RS_T_300000.nnue /home/user/nets/p6/RS_G_300000.nnue 500
echo P7_DONE
