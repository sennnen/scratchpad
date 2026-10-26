#!/bin/bash
# E7: (a) quantized incremental student == float student?  (b) teacher gap at equal nodes,
# expressed as teacher node odds;  (c) equal-time match (10+0.1) of student vs teacher.
BEST=${BEST:-/home/user/nets/e2/base_T1S1.stud}
cd /home/user/scratchpad; R=experiments/e7/matches
Q=/home/user/data/sf_research_kb; F=/home/user/data/sf_research
run() { out=$1; shift; [ -f $R/$out.json ] || python3 tools/match.py "$@" --out $R/$out | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
run quant_vs_float sq=$BEST sf=$BEST --bin_a $Q --bin_b $F --nodes 20000 --rounds 150
run stud_vs_teacher_n20k stud=$BEST teacher=teacher --bin_a $Q --bin_b $Q --nodes 20000 --rounds 50
run stud_vs_teacher_n5k stud=$BEST teacher=teacher@5000 --bin_a $Q --bin_b $Q --nodes 20000 --rounds 75
run stud_vs_teacher_n1250 stud=$BEST teacher=teacher@1250 --bin_a $Q --bin_b $Q --nodes 20000 --rounds 75
run stud_vs_teacher_tc10 stud=$BEST teacher=teacher --bin_a $Q --bin_b $Q --tc 10+0.1 --rounds 50
echo E7_DONE
