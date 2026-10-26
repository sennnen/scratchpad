#!/bin/bash
# E2b: more games for the decisive head-to-head (independent opening seed)
cd /home/user/scratchpad; N=/home/user/nets/e2
[ -f experiments/e2/matches/base_T1S1_vs_base_T1T2_s2.json ] || python3 tools/match.py base_T1S1=$N/base_T1S1.stud base_T1T2=$N/base_T1T2.stud --nodes 20000 --rounds 600 --seed 2 --out experiments/e2/matches/base_T1S1_vs_base_T1T2_s2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"
echo E2B_DONE
