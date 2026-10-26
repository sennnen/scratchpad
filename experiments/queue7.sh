#!/bin/bash
until grep -q E7_DONE /home/user/nets/e7.log 2>/dev/null; do sleep 30; done
/home/user/scratchpad/experiments/e9/run.sh > /home/user/nets/e9.log 2>&1
