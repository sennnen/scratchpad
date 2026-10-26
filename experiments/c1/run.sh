#!/bin/bash
# C1: iterated on-policy (DAgger) distillation, H=256 small students (Phase I setting, float binary).
# Iteration 1 = E2: ON_1 = T1+S1 (student = E1 tree_teacher), OFF_1 = T1+T2.
# Iteration k>1: S_k = self-play of ON_{k-1}'s student (teacher-labelled); T_{k+1} = teacher self-play,
# same games/seeds. ON_k = ON_{k-1} + S_k, OFF_k = OFF_{k-1} + T_{k+1}. 1500 games x 8% per new block.
until grep -q B2_DONE /home/user/nets/b2.log 2>/dev/null; do sleep 30; done
G=/home/user/data/sf_research; B=/home/user/data/UHO_Lichess_4852_v1.epd; D=/home/user/data/c1; N=/home/user/nets/c1; mkdir -p $D $N
VAL="--val vgame=/home/user/data/val/v.bin:0 --val vtree=/home/user/data/val/v.bin:1 --val vstree=/home/user/data/val/vs.bin:1"
T=/home/user/nets/trainer
gen() { tag=$1; stud=$2; seedbase=$3
  for i in 1 2 3 4; do
    [ -f $D/$tag.$i.done ] || { (if [ -n "$stud" ]; then echo "setoption name StudentFile value $stud"; fi
      echo "gendata nodes 5000 games 375 out $D/$tag.$i.bin book $B seed $((seedbase+i)) treerate 0.08"; echo quit) | $G 2>&1 | grep -a "gendata games" | tail -n1 > $D/$tag.$i.log; touch $D/$tag.$i.done; } &
  done; wait; cat $D/$tag.*.log; }
files() { for i in 1 2 3 4; do echo --train $1.$i.bin; done; }
R1=$(for i in 1 2 3 4; do echo --train /home/user/data/run1/d$i.bin; done)
ON="$R1 $(files /home/user/data/e2/S1)"; OFF="$R1 $(files /home/user/data/e2/T2)"
cp /home/user/nets/e2/base_T1S1.stud $N/ON_1.stud; cp /home/user/nets/e2/base_T1T2.stud $N/OFF_1.stud
cd /home/user/scratchpad; R=experiments/c1/matches
m() { [ -f $R/$1_vs_$2.json ] || python3 tools/match.py $1=$N/$1.stud $2=$N/$2.stud --nodes 20000 --rounds $3 --seed $4 --out $R/$1_vs_$2 | tail -1 | sed "s/'cmd'[^]]*\],//; s/=[^ ]*stud//g"; }
for k in 2 3; do
  prev=$((k-1))
  gen S$k $N/ON_$prev.stud $((12000+100*k))
  gen Tc$k "" $((12000+100*k))
  ON="$ON $(files $D/S$k)"; OFF="$OFF $(files $D/Tc$k)"
  [ -f $N/ON_$k.stud ] || $T $ON $VAL --src 1 --positions 2e8 --out $N/ON_$k.stud 2> $N/ON_$k.log
  [ -f $N/OFF_$k.stud ] || $T $OFF $VAL --src 1 --positions 2e8 --out $N/OFF_$k.stud 2> $N/OFF_$k.log
  for a in ON_$k OFF_$k; do grep val $N/$a.log | awk '{printf "%s %s ", $2, $4}'; echo "<- $a"; done
  m ON_$k OFF_$k 500 $k
  m ON_$k ON_$prev 300 $k
  m OFF_$k OFF_$prev 300 $k
done
echo C1_DONE
