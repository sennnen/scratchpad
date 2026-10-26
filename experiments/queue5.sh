#!/bin/bash
until grep -q E6_DONE /home/user/nets/e6.log 2>/dev/null; do sleep 30; done
/home/user/scratchpad/experiments/e2/more.sh > /home/user/nets/e2b.log 2>&1
