#!/bin/bash
cd /home/user/scratchpad/experiments
./e3/run.sh > /home/user/nets/e3.log 2>&1
echo QUEUE_DONE >> /home/user/nets/e3.log
