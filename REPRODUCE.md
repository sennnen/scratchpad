# Reproduce

All experiments ran on a 4-vCPU Xeon (AVX-512 + VNNI, no VBMI2), 15 GB RAM, no GPU.

## 1. Pinned sources
```bash
git clone https://github.com/official-stockfish/Stockfish && cd Stockfish
git checkout ff3a6c508ed731f0dcf253f4ddf646df19a62f84
git am /path/to/this/repo/sf-patches/0001-research-gendata-student.patch
```
Network `nn-c9e480e92425.nnue` (sha256 c9e480e92425f887df6ab9b05c6180fbd762b07879df9d9845f1aa75440caa73).
If tests.stockfishchess.org / Git LFS are unreachable, extract it from the matching prerelease binary:
```bash
curl -LO https://github.com/official-stockfish/Stockfish/releases/download/stockfish-dev-20261025-ff3a6c50/stockfish-linux-x86-64-universal.tar.gz
tar xzf stockfish-linux-x86-64-universal.tar.gz
printf "uci\nexport_net nn-c9e480e92425.nnue\nquit\n" | ./stockfish/stockfish-linux-x86-64-universal
sha256sum nn-c9e480e92425.nnue
```
Build: `cd src && cp ../../nn-c9e480e92425.nnue . && make -j4 profile-build ARCH=x86-64-vnni512 COMP=gcc`
(bench must print 2389370 nodes: the patch is not a functional change unless StudentFile/gendata are used).

fastchess `ebfd64cb7fac80d4822c50b07dca4fe934779d21` (`make -j4`). Books from
https://github.com/official-stockfish/books: `UHO_Lichess_4852_v1.epd` (data generation),
`UHO_4060_v4.epd` (matches; disjoint from training openings).

Trainer: `g++ -O3 -march=native -ffast-math -std=c++20 -pthread tools/trainer/train.cpp -o trainer`.
(E1-E3 ran an earlier build of the same file before `--fset`/`--skip`/reserve were added; the
feature/forward/backward code for `--fset 0` is unchanged. Fork patches: apply 0001-0004 in order;
E1-E3 matches used the float evaluator of patch 0001, E4+ the incremental int16 evaluator of 0002/0003.)

## 2. Data generation (UCI command added by the patch)
```
gendata nodes 5000 games 1500 out d1.bin book UHO_Lichess_4852_v1.epd seed 101 treerate 0.02
```
Run 4 processes with seeds 101..104 (run1). Validation: 60 games/proc, seeds 901..904.
Optional `setoption name StudentFile value net.stud` before `gendata` makes the *student* search
(on-policy data); recorded positions are still labelled by the teacher network.
Record format: `tools/data.py` / `src/research.h` (40 bytes).

## 3. Training
```
trainer --train d1.bin ... --src 1 --label teacher --positions 1e8 --out tree_teacher.stud \
        --val vgame=v.bin:0 --val vtree=v.bin:1
```
`--src 0` game positions, `--src 1` search-tree positions; `--label teacher|score|blend --lam 0.7`;
`--frac f` uses the first fraction f of each file (= fraction of games/generation compute);
`--n k` random subsample of k positions. Experiment scripts: `experiments/*/run*.sh`.

## 4. Matches and ratings
```
python3 tools/match.py A=a.stud B=b.stud --nodes 20000 --rounds 100 --out experiments/x/matches/A_vs_B
python3 tools/ratings.py experiments/e1/matches --anchor game_teacher_p1.5e7
```

## Phase II (production-architecture experiments)
Fork patches 0005-0008 on top of 0001-0004: `ftdump` (FT outputs + bucket + eval, SEE flag), `label`
capture flag, ply-from-root and search-context (beta, PV, qsearch) metadata for tree records.
Python deps: numpy, torch (CPU). Scripts: `experiments/p2_label.sh`, `experiments/p2f1/run.sh`,
`experiments/p3_run.sh` ... `p8_run.sh`, `experiments/b1|b2|b4|c1/run.sh`.

```bash
# label positions with an N-node search and dump FT outputs
experiments/p2_label.sh 5000 VG VT G T            # needs tools/p2/make_sets.py output first
# fine-tune / re-distill the 32 layer stacks of the production net
python3 tools/p2/finetune.py --net nn-c9e480e92425.nnue --train T.ft --val vg=VG.l5000.ft vt=VT.l5000.ft \
   --target teacher --reinit --lr 3e-3 --steps 3000 --out RS_T.nnue      # P3 recipe
python3 tools/p2/finetune.py ... --target search --quiet --nkeep 186365 --lr 1e-4 --steps 1500   # P2-F1
# verify an exported net against Stockfish's integer inference
(echo "setoption name EvalFile value RS_T.nnue"; echo "ftdump in VG.l5000.bin out check.ft max 3000"; echo quit) | ./stockfish
python3 tools/p2/verify_float.py RS_T.nnue check.ft      # expect max |diff| < 1 internal unit
# matches with full nets
python3 tools/match.py A=RS_T.nnue master=teacher --nodes 20000 --rounds 500 --out experiments/x/A_vs_master
```
Important: `Rec.score` written by `gendata`/`label` is in **centipawns** (Stockfish `to_cp`); convert with
`tools/p2/finetune.py:cp_to_internal` before using it as a target in internal units.
