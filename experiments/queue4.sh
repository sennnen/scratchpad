#!/bin/bash
until grep -q E4_DONE /home/user/nets/e4.log 2>/dev/null; do sleep 30; done
/home/user/scratchpad/experiments/e6/run.sh > /home/user/nets/e6.log 2>&1
