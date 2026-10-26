#!/bin/bash
until grep -q E2B_DONE /home/user/nets/e2b.log 2>/dev/null; do sleep 30; done
NAME=e8_h1024_all ARCH="--H 1024" P=4e8 /home/user/scratchpad/experiments/e8/run.sh > /home/user/nets/e8.log 2>&1
echo /home/user/nets/e8/e8_h1024_all.stud > /home/user/nets/best_student
BEST=/home/user/nets/e8/e8_h1024_all.stud /home/user/scratchpad/experiments/e7/run.sh > /home/user/nets/e7.log 2>&1
