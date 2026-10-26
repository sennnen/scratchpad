#!/bin/bash
# E8 (escalation): best E4 architecture trained on all tree data collected in this session
# (teacher trees T1, T2, E3 runs; student trees S1). Then E7-style evaluation vs teacher.
# Usage: ARCH="--H 1024" or ARCH="--H 512 --fset 1"; NAME=e8_xxx
T=/home/user/nets/trainer.kb; O=/home/user/nets/e8; mkdir -p $O
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1 --val vstree=/home/user/data/val/vs.bin:1"
FILES=""
for f in /home/user/data/run1/d{1,2,3,4}.bin /home/user/data/e2/T2.{1,2,3,4}.bin /home/user/data/e2/S1.{1,2,3,4}.bin \
         /home/user/data/e3/n5000_r08.{1,2,3,4}.bin /home/user/data/e3/n20000_r08.{1,2,3,4}.bin /home/user/data/e3/n1000_r08.{1,2,3,4}.bin; do
  FILES="$FILES --train $f"; done
[ -f $O/$NAME.stud ] || $T $FILES $VAL --src 1 $ARCH --positions ${P:-4e8} --out $O/$NAME.stud 2> $O/$NAME.log
grep -E "val|wall" $O/$NAME.log
echo E8_TRAIN_DONE
