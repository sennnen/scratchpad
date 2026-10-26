#!/bin/bash
# sequential experiment queue
cd /home/user/scratchpad/experiments
until grep -q E1B_DONE /home/user/nets/e1b.log; do sleep 20; done
./e2/controlled.sh > /home/user/nets/e2.log 2>&1
./e3/run.sh > /home/user/nets/e3.log 2>&1
echo QUEUE_DONE >> /home/user/nets/e3.log
