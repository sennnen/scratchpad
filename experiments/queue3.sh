#!/bin/bash
until grep -q E1C_DONE /home/user/nets/e1c.log 2>/dev/null; do sleep 30; done
/home/user/scratchpad/experiments/e4/run.sh > /home/user/nets/e4.log 2>&1
