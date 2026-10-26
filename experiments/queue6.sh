#!/bin/bash
until grep -q E2B_DONE /home/user/nets/e2b.log 2>/dev/null; do sleep 30; done
BEST=$(cat /home/user/nets/best_student 2>/dev/null || echo /home/user/nets/e2/base_T1S1.stud) /home/user/scratchpad/experiments/e7/run.sh > /home/user/nets/e7.log 2>&1
