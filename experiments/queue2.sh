#!/bin/bash
until grep -q QUEUE_DONE /home/user/nets/e3.log 2>/dev/null; do sleep 30; done
/home/user/scratchpad/experiments/e1c/run.sh > /home/user/nets/e1c.log 2>&1
