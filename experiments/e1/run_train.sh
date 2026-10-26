#!/bin/bash
# E1: label source / position source study. All students H=256, NB=8.
# Usage: run_train.sh  (expects run1 data and val data in /home/user/data)
set -e
T=/home/user/nets/trainer
D=/home/user/data/run1
O=/home/user/nets/e1
mkdir -p $O
TR="--train $D/d1.bin --train $D/d2.bin --train $D/d3.bin --train $D/d4.bin"
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1"
P=${P:-1e8}
run() { name=$1; shift; [ -f $O/$name.stud ] || $T $TR $VAL --out $O/$name.stud --positions $P "$@" 2> $O/$name.log; tail -n 3 $O/$name.log; }
# game positions, teacher static-eval labels
run game_teacher --src 0 --label teacher
# game positions, conventional labels: 0.7*search score + 0.3*game result (5000-node search)
run game_blend --src 0 --label blend --lam 0.7
# game positions, pure search score
run game_score --src 0 --label score
# tree positions, sample-matched to the game set size
run tree_teacher_sm --src 1 --label teacher --n $(cat $O/n_game 2>/dev/null || echo 730000)
# tree positions, all (same generation compute as the game set)
run tree_teacher --src 1 --label teacher
