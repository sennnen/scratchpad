#!/bin/bash
# Gauntlet: every E1 student vs reference student game_blend (conventional labels), paired openings.
N=/home/user/nets/e1
R=/home/user/scratchpad/experiments/e1/matches
mkdir -p $R
ROUNDS=${ROUNDS:-500}
NODES=${NODES:-20000}
REF=game_blend
for s in game_teacher game_score tree_teacher_sm tree_teacher; do
  [ -f $R/${s}_vs_$REF.json ] || python3 /home/user/scratchpad/tools/match.py $s=$N/$s.stud $REF=$N/$REF.stud --nodes $NODES --rounds $ROUNDS --out $R/${s}_vs_$REF
done
echo MATCHES_DONE
