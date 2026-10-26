# Research log

## 2026-10-25/26 Session 1: environment and baselines

### Environment (measured)
- CPU: 4 vCPU Intel Xeon @ 2.80GHz, AVX-512 F/DQ/CD/BW/VL + VNNI (no VBMI2/IFMA, so
  `x86-64-avx512icl` builds SIGILL; use `x86-64-vnni512`). No GPU.
- RAM 15 GB, no swap. Disk ~31 GB free. Network: git reads of public GitHub repos,
  GitHub release assets, PyPI. Blocked: tests.stockfishchess.org, huggingface.co,
  download.pytorch.org, Git LFS objects (official-stockfish/networks is LFS).
- Workaround for nets: download official release/prerelease binaries and `export_net`
  the embedded network; verify SHA-256 prefix equals the file name.

### Pinned upstream revisions
- Stockfish master `ff3a6c508ed731f0dcf253f4ddf646df19a62f84` (2026-10-25, "21 bit tt keys with SWAR"),
  net `nn-c9e480e92425.nnue` (SFNNv18: K32Q2 + FullThreats + PP_3Wide, L1=1024, 32 layer stacks), bench 2389370.
  Net extracted from release `stockfish-dev-20261025-ff3a6c50` (linux universal), sha256 c9e480e92425f887...
- Stockfish latest stable `sf_19` (2026-09-05), net `nn-1a298aa575a0.nnue` (sha256 verified).
- nnue-pytorch `ed84cc1fd2255faa0f33dcfa7839c9b6b4bb49b8`; fishtest `5a8a229a521b4eaab2ecf19b92f0d2fd6c3be13b`.
- fastchess `ebfd64cb7fac80d4822c50b07dca4fe934779d21`. books: official-stockfish/books (shallow, 2026-10-25).

### Baseline speed
- Master PGO build (`make profile-build ARCH=x86-64-vnni512 COMP=gcc`): bench 2389370 nodes, ~480-495k nps, 1 thread.

### Strategic assessment (why this research direction)
- Full-strength SPRT testing of search patches is out of reach: at 10+0.1 we can play
  ~800 games/hour on 4 cores; typical Stockfish gains (1-3 Elo) need 30k-200k games.
- Full-scale NNUE training (billions of Leela positions, GPU) is out of reach, and Leela
  data is not downloadable here.
- What *is* cheap here: teacher evaluations. The master net evaluates ~170k positions/s/core
  inside search. So the research question chosen is: **how cheaply can playing strength be
  transferred into a new network, and which data/label sources give the most Elo per CPU-second?**
- Key observation motivating H1: every `evaluate()` call during a teacher search is a
  (position, teacher eval) pair at zero extra cost, and its distribution is exactly the
  distribution on which an evaluation function is used during search. Game positions
  (the conventional training distribution) are a tiny, biased subset (quiet, well-played).

### Hypotheses
- H1: Students distilled from teacher search-tree positions are stronger (per training sample and per
  CPU-second) than students distilled from game positions.
- H2: Teacher static-eval labels beat conventional search-score+WDL labels at matched label compute.
- H3 (DAgger): positions from the *student's own* search trees, labelled by the teacher, fix the
  covariate shift between training and use, further improving strength per sample.

### Infrastructure built
- Stockfish fork (`sf-patches/`): `gendata` UCI command (self-play at N nodes from an EPD book,
  writes 40-byte records for game positions with search score/teacher eval/result, and a random
  fraction of evaluate() calls with teacher labels); `StudentFile` option replaces the NNUE eval
  by a float student (768->H)x2 SCReLU -> 8 buckets, keeping Stockfish's eval scaling and search.
  With no StudentFile and no recorder, bench is unchanged (2389370): no functional change.
- Verified: engine student eval == independent numpy forward (max abs diff 0 on 1084 positions);
  C++ trainer validation MAE == numpy (311.27 both).
- C++ trainer: ~225k positions/s on 2 contended threads vs ~12k/s for a PyTorch EmbeddingBag
  implementation on the same CPU (PyTorch abandoned for this net size).

### Data run1 (E1 training data)
- `gendata nodes 5000 games 1500 treerate 0.02`, seeds 101-104, book UHO_Lichess_4852_v1 (random line per game), 4 processes.
- 6000 games (W/D/L 1985/2584/1431 from white), 734,092 game positions, 25,145,952 tree positions, 1.04 GB.
- Cost: ~1872 s wall per process, 4 processes = ~7,490 CPU-s total (game + tree data produced by the same searches).
  Game positions: ~98 /CPU-s. Tree positions (2% of evaluate() calls): ~3,360 /CPU-s; 100% would be ~168k /CPU-s.
- Validation: separate run (seeds 901-904, 240 games) with the same settings -> `val/v.bin`.
- Matches use a different book (UHO_4060_v4) to avoid opening overlap with training games.
- Ops lesson: `pkill -f`/`pgrep -f` patterns matched the harness's own shells (whose command lines contain the
  pattern); use `pgrep -x <procname>` and never put the kill pattern in the same command line.

## E1: position source and label source (H=256 students, 20k nodes/move matches)
Setup: all students (768->256)x2 SCReLU -> 8 buckets, Adam, one-cycle cosine, batch 16384, trained on
run1 (7,490 CPU-s of 5000-node self-play). Validation: held-out games (seeds 901-904).
Matches: research fork with StudentFile, `nodes=20000`, UHO_4060_v4 openings (disjoint book), paired,
200 games per pair; joint Bradley-Terry fit with pentanomial bootstrap (`tools/ratings.py`).

Validation MSE vs teacher (sigmoid space, scale 340):
| student | positions trained | vgame | vtree |
|---|---|---|---|
| game_teacher (P=1e8, overfit) | 734k | 0.0311 | 0.0406 |
| game_teacher_p1.5e7 (best length) | 734k | 0.0249 | 0.0344 |
| gq_teacher_p1.5e7 (quiet filter) | 594k | 0.0254 | 0.0347 |
| tree_teacher_sm (sample-matched) | 734k | 0.0289 | 0.0296 |
| tree_teacher | 25.1M | 0.0186 | 0.0186 |

Elo (anchor game_teacher_p1.5e7): tree_teacher +539 [472,626]; tree_teacher_sm +205 [156,257];
gq_teacher -22 [-67,21]; gq_score -113 [-161,-63]; gq_blend(0.7) -252 [-312,-208];
unfiltered game_blend/score: -600 to -1300 (static evals trained on unfiltered search scores are badly miscalibrated).

Observations:
- At equal sample count, tree positions give +205 Elo over game positions, although their MSE on *game*
  validation positions is worse (0.0289 vs 0.0249). Validation loss on game positions is a misleading
  proxy for strength; loss on the distribution the evaluator sees in search (vtree) ranks students correctly.
- Weak students search far shallower at equal nodes (depth 8-9 vs 12-16) and hang material: eval errors
  on off-distribution (material-imbalanced) positions degrade pruning, not just leaf accuracy.
- Teacher static labels beat search-score labels (gq_teacher vs gq_score: +91), and WDL blending with
  5000-node self-play results hurts (-139 vs pure score) at this data scale.

## E1b: data scaling (generation-compute fractions of run1), 20k nodes, joint fit with E1
`--frac f` = first fraction f of each generation file = fraction f of the games = fraction f of the
7,490 generation CPU-s. Training length: tree 1.6e7..1e8 positions, game ~20 epochs (best on validation).
Joint Bradley-Terry fit (`experiments/e1b/ratings_joint.txt`, aliases game_f1=game_teacher_p1.5e7,
tree_f1=tree_teacher), anchor game_teacher_p1.5e7 = 0:

| gen. compute | tree-position student | game-position student (teacher labels) |
|---|---|---|
| 1/32 (234 CPU-s) | +111 [75,151] | |
| 1/16 | +264 [214,311] | |
| 1/8 | +398 [353,448] | |
| 1/4 | +465 [413,516] | -362 [-430,-303] |
| 1/2 | +520 [466,576] | -159 [-202,-113] |
| 1 (7,490 CPU-s) | +545 [493,615] | 0 |
Conventional baseline at full compute (quiet positions, 5000-node search-score labels): -113 [-165,-66].

Direct check: tree_f32 vs game_f1: +104 +/- 49 (200 games, LOS 100%).
=> A student distilled from teacher search-tree positions with 1/32 of the generation compute is
stronger than any game-position student trained on all of it. Total CPU (generation + training):
tree_f32 ~ 234 + ~70 = ~300 CPU-s vs game_f1 ~ 7,556 CPU-s: >= 25x less total compute, and +111 Elo.
The tree curve flattens (+69, +56, +26 per doubling from 1/8 up), consistent with the 256-neuron
student nearing capacity; the game curve is still steep (+203, +159 per doubling) but starts far lower.
Caveats: small student (768->256), node-limited matches (equal nodes, not equal time), single teacher,
one generation setting (5000 nodes/move). Distillation requires an existing strong net.

### Validation-loss proxy for strength (E1+E1b, 11 teacher-label H=256 students)
`tools/proxy_corr.py`: Elo vs held-out MSE. Search-tree validation MSE: Pearson 0.996, Spearman 1.000.
Game-position validation MSE: Pearson 0.850, Spearman 0.800; it ranks game_teacher_p1.5e7 (vgame 0.0249)
above tree_f32/tree_teacher_sm/tree_f16 (vgame 0.026-0.029), which beat it by 111-264 Elo.
=> Selecting/validating nets on game positions (common practice) is misleading for students trained on
different distributions; loss on positions sampled from search trees is a near-perfect cheap proxy here.
(Within one architecture and teacher; untested across architectures.)

### Student inference engineering (patch 0002)
- Float full-refresh student: 365k nps vs teacher 398k (same contended conditions): no speed benefit.
- New: per-thread int16 accumulators for WHITE/BLACK perspectives, updated by diffing the 64-square
  board against the last evaluated position (no do_move hooks; refresh if >12 squares differ);
  float output layer. QA chosen per net so the worst case cannot overflow int16 (QA~700 here).
  Accuracy vs float reference on 5000 search positions: mean |diff| 2.1, max 8 internal units.
  Speed: ~560-780k nps under heavy contention (~2x the float student). Binary: sf_research_q.
  Running experiments (E2/E3/E1c/E4) keep the float binary for consistency.

## E2: on-policy (DAgger-style) data vs more teacher-tree data
Data: S1 = 6000 self-play games where the *student* (E1 tree_teacher, H=256) searches (5000 nodes/move),
2% of its evaluate() calls recorded with teacher labels: 24.1M positions, ~7,730 CPU-s (student search
was not faster than the teacher's with the float evaluator). T2 = 6000 more teacher self-play games
(seeds 201-204): 25.3M positions, ~7,580 CPU-s. Student self-play is more decisive (W/D/L from white
2371/1611/2018 vs teacher 1973/2641/1386).
Distribution stats (1M samples): share with material imbalance >= 300: game 0.241, teacher-tree 0.318,
student-tree 0.341; median |teacher eval| 317 / 506 / 788 internal units.
New validation set vs.bin: student (tree_teacher) self-play trees, seeds 951-954.
All H=256, 2e8 training positions:
| data | vgame | vtree | vstree |
|---|---|---|---|
| T1 (25M) | 0.01721 | 0.01702 | 0.01741 |
| T1+T2 (50M) | 0.01633 | 0.01658 | 0.01662 |
| T1+S1 (49M) | 0.01667 | 0.01706 | 0.01583 |
Note: T1 trained for 2e8 instead of 1e8 positions improves vtree 0.01863 -> 0.01702, so E1b's fixed
budgets understate the large-data points (the efficiency claim R1 is conservative in that respect).
