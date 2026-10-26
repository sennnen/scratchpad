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
