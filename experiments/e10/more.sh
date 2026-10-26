#!/bin/bash
until grep -q E10_MATCHES1_DONE /home/user/nets/e10.log; do sleep 20; done
N=/home/user/nets/e10; cd /home/user/scratchpad; R=experiments/e10/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds ${3:-100} --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m game_n100_p3e7 game_n1000_p1.5e7
m tree_f32 game_n100_p3e7 150
echo E10_DONE
