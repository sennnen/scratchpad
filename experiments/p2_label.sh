#!/bin/bash
# Phase II: label sets with an N-node search, then dump FT outputs. Usage: p2_label.sh NODES SET...
N=$1; shift; D=/home/user/data/p2; B=/home/user/data/sf_p2
for S in "$@"; do
  for k in 0 1 2 3; do
    [ -f $D/$S.l$N.$k.done ] || { (echo "label in $D/$S.in.$k.bin out $D/$S.l$N.$k.bin nodes $N"; echo quit) | $B > /dev/null 2>&1; touch $D/$S.l$N.$k.done; } &
  done; wait
  cat $D/$S.l$N.{0,1,2,3}.bin > $D/$S.l$N.bin
  [ -f $D/$S.l$N.ft ] || { (echo "ftdump in $D/$S.l$N.bin out $D/$S.l$N.ft"; echo quit) | $B | grep ftdump; }
done
echo LABEL_DONE
