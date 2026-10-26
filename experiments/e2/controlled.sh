#!/bin/bash
# E2 controlled: does on-policy (student-search) data beat equal amounts of teacher-search data?
G=/home/user/data/sf_research; B=/home/user/data/UHO_Lichess_4852_v1.epd
D=/home/user/data/e2; N=/home/user/nets/e2; mkdir -p $D $N
S0=/home/user/nets/e1/tree_teacher.stud
gen() { tag=$1; stud=$2; games=$3; seedbase=$4; out=$5
  for i in 1 2 3 4; do
    [ -f $out/$tag.$i.done ] && continue
    { (if [ -n "$stud" ]; then echo "setoption name StudentFile value $stud"; fi
       echo "gendata nodes 5000 games $games out $out/$tag.$i.bin book $B seed $((seedbase+i)) treerate 0.02"; echo quit) \
       | $G 2>&1 | grep -a "gendata games" | tail -n1 > $out/$tag.$i.log; touch $out/$tag.$i.done; } &
  done; wait; cat $out/$tag.*.log; }
mkdir -p /home/user/data/val
gen vs "$S0" 60 950 /home/user/data/val
cat /home/user/data/val/vs.{1,2,3,4}.bin > /home/user/data/val/vs.bin
gen S1 "$S0" 1500 300 $D
gen T2 "" 1500 200 $D
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1 --val vstree=/home/user/data/val/vs.bin:1"
R1="--train /home/user/data/run1/d1.bin --train /home/user/data/run1/d2.bin --train /home/user/data/run1/d3.bin --train /home/user/data/run1/d4.bin"
tr() { name=$1; shift; [ -f $N/$name.stud ] || /home/user/nets/trainer $VAL --src 1 --positions 2e8 --out $N/$name.stud "$@" 2> $N/$name.log; grep -E "val|wall" $N/$name.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $name"; }
tr base_T1T2 $R1 $(for i in 1 2 3 4; do echo --train $D/T2.$i.bin; done)
tr base_T1S1 $R1 $(for i in 1 2 3 4; do echo --train $D/S1.$i.bin; done)
tr base_T1_2e8 $R1
echo E2_TRAIN_DONE
cd /home/user/scratchpad; mkdir -p experiments/e2/matches
m() { [ -f experiments/e2/matches/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds ${3:-200} --out experiments/e2/matches/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
m base_T1S1 base_T1T2 300
m base_T1T2 base_T1_2e8 150
m base_T1S1 base_T1_2e8 150
echo E2_DONE
