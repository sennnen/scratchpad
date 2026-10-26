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
Matches (20k nodes, UHO_4060_v4, paired):
- base_T1S1 vs base_T1T2: +13.9 +/- 23.1 (600 games, ptnml [35,45,125,51,44], LOS 88%)  -> inconclusive
- base_T1T2 vs base_T1_2e8: +2.3 +/- 34.7 (300 games)  -> no measurable gain from 2x teacher-tree data
- base_T1S1 vs base_T1_2e8: +46.6 +/- 32.2 (300 games, LOS 99.8%)
Joint fit (anchor base_T1_2e8): T1S1 +34 [8, 58], T1T2 +14 [-10, 41].
Interpretation (preliminary): at H=256 the student saturates on teacher-tree data (loss improves,
Elo does not), while adding its own search-tree positions (teacher-labelled) still adds strength:
consistent with H3 (covariate shift between teacher and student search distributions). The decisive
head-to-head is not yet significant; 1,200 more games queued (E2b, opening seed 2).

### Ops incident (E3)
A helper function named `tr` piped its output into coreutils `tr`, recursing into itself and spawning
thousands of nested shells (~3 minutes, killed). No data was lost; E3 generation had finished and is
unaffected; training was relaunched. Rule: never shadow common command names in experiment scripts.

## E3: generation settings at an equal generation budget (~1,870 CPU-s = 4 procs x 468 s)
Reference n5000_r02 = e1b tree_f4 (5000 nodes/move, 2% of evaluate() calls; same compute). All H=256,
1e8 training positions. 20k-node matches vs reference, 200 games each:
| setting | records | vgame | vtree | Elo vs ref |
|---|---|---|---|---|
| 1000 nodes, 2% | 5.0M | 0.02241 | 0.02336 | -38 +/- 40 |
| 5000 nodes, 2% (ref) | 6.3M | 0.02115 | 0.02179 | 0 |
| 1000 nodes, 8% | 17.5M | 0.02077 | 0.02187 | +2 +/- 39 |
| 5000 nodes, 8% | 25.3M | 0.02071 | 0.02029 | +33 +/- 44 |
| 20000 nodes, 8% | 28.7M | 0.02050 | 0.02016 | +51 +/- 38 (LOS 99.6%) |
- Recording more of each search's evaluate() calls is nearly free and helps; deeper per-move searches
  are at least as good as shallow ones per CPU-s (1000 nodes/move: per-move overhead dominates, fewer evals/s).
- Elo ordering again equals the vtree-MSE ordering (R2 holds out of sample).
- Implication for R1: the 2%-rate setting used there is not the most efficient; the >=32x figure is conservative.

## E1c: replication and node-budget robustness of R1
Disjoint 1/32 subsets of run1 (`--frac 0.03125 --skip s`), H=256, 1.6e7 training positions; opponent:
game-position student trained on all of run1 (game_f1). 200 games each, 20k nodes:
| subset | vtree | Elo vs game_f1 |
|---|---|---|
| s=0 (E1b tree_f32) | 0.02975 | +104 +/- 49 |
| s=0.25 | 0.03150 | +147 +/- 47 |
| s=0.50 | 0.02984 | +133 +/- 47 |
| s=0.75 | 0.03067 | +81 +/- 46 |
4/4 subsets significant (LOS >= 99.99%); mean +116. Node budgets (tree_f32 vs game_f1):
5k nodes +123 +/- 52, 20k +104 +/- 49, 80k +141 +/- 50. Full data at 80k nodes (tree_f1 vs game_f1):
193-2-5, +655 +/- 176 (vs +545 at 20k): the advantage does not shrink with deeper search.

## E4: capacity scaling on the same run1 data (7,490 generation CPU-s), 20k nodes
Tree students 1.5e8 training positions; game students ~20 epochs. Matches on the quantized incremental
binary (all E4 games); E1/E1b used the float binary (neutrality check in E7).
| student | vgame | vtree | vstree | match |
|---|---|---|---|---|
| tree H=512 | 0.01641 | 0.01643 | 0.01623 | vs H256: +70 +/- 38 |
| tree H=1024 | 0.01557 | 0.01542 | 0.01531 | vs H512: +72 +/- 39 |
| tree kb256 (8 king buckets, mirrored, factorised) | 0.01638 | 0.01675 | 0.01644 | vs H256: +58 +/- 39 |
| tree kb512 | 0.01565 | 0.01570 | 0.01506 | vs kb256: +47 +/- 36; vs H1024: -26 +/- 38 |
| game H=512 | 0.02582 | 0.03516 | 0.03651 | vs game H256: +2 +/- 39 |
| game H=1024 | 0.02650 | 0.03535 | 0.03734 | vs game H512: -7 +/- 46 |
| game kb256 | 0.02925 | 0.03939 | 0.04136 | vs game H256: -133 +/- 44 |
Joint ladder (experiments/e4/ratings_joint.txt, anchor game H256 = 0): tree H1024 +683 [616,751],
tree kb512 +654, tree H512 +613, tree kb256 +605, tree H256 +545; game H512 +2, game H1024 -5,
game kb256 -133; best conventional-label student -113.
=> With game positions, extra capacity or richer features buy nothing (or hurt: overfitting to ~734k
correlated positions). With tree positions every width doubling adds ~+70 Elo. The data-source gap
widens with model size: ~+680 Elo between the best students of each kind at equal generation compute.

## E6: without distillation (search-score labels on both sides)
600k positions sampled uniformly from run1 search trees, each re-labelled by a fresh 5000-node search
(`label` command, ~6,000 CPU-s, castling/ep rights dropped) vs the 594k quiet game positions of run1 whose
labels are the 5000-node in-game search scores (gq_score_p1.5e7). Same label type, count and cost per label.
H=256, `--label score`.
| student | vgame (vs teacher) | vtree (vs teacher) | Elo vs gq_score_p1.5e7 |
|---|---|---|---|
| tree_score_p1.5e7 | 0.05029 | 0.04627 | +80 +/- 36 (300 games, LOS 100%) |
| tree_score_p5e6 | 0.05197 | 0.04751 | -9 +/- 35 |
=> The position-distribution effect is not specific to distillation: sampling training positions from
search trees beats (quiet-filtered) game positions even with conventional search labels.
Note: MSE-vs-teacher is not a valid proxy across different label types (tree_score has worse MSE but is
stronger); R2 applies to students trained on the same target.

### E2b: confirmation of the on-policy effect
base_T1S1 vs base_T1T2, 1,200 more games with an independent opening seed: +27.0 +/- 15.8 (LOS 99.96%).
Pooled with the first 600 games (1,800 games, 900 pairs): **+22.6 [9.6, 35.7]**. Joint fit (anchor T1):
T1S1 +36.5 [11.7, 58.9], T1T2 +12.3 [-11.3, 36.0].
=> H3 supported: at equal data volume and generation cost, teacher-labelled positions from the
student's *own* search trees beat positions from the teacher's search trees by ~+23 Elo (H=256).
Single iteration only; whether further DAgger iterations keep paying is untested.

## E8: escalation (H=1024, all session tree data)
H=1024 (768 features) trained on all tree positions generated this session: run1 T1, T2 (teacher trees),
S1 (student trees), E3 n1000_r08/n5000_r08/n20000_r08: ~144M positions, 4e8 training positions
(2,251 s wall, 7,625 CPU-s, peak RSS ~6.3 GB). Validation: vgame 0.01192, vtree 0.01222, vstree 0.01113
(vs 0.01557/0.01542/0.01531 for H=1024 on run1 only). Total generation compute behind the data:
~7,490 (T1) + 7,580 (T2) + 7,730 (S1) + 3 x 1,872 (E3) = ~28,400 CPU-s (~7.9 CPU-hours).

## E7: teacher gap of the E8 student (quantized incremental binary for both sides)
- Quantized int16 incremental vs float student (same net, 20k nodes): +9.3 +/- 28.6 (300 games): neutral.
- Student @20k nodes vs teacher @20k: -382 +/- 80 (100 games, 1-81-18).
- Student @20k vs teacher @5k: -56 +/- 45 (150 games). Student @20k vs teacher @1.25k: +409 +/- 82.
  => the student is worth a teacher searching ~1/5 of the nodes (teacher gains +465 from 1.25k->5k and
  +326 from 5k->20k nodes in this ladder).
- Equal time, 10+0.1, 1 thread: -330 +/- 54 (100 games, 0-74-26). Speed (8 s search, contended machine):
  teacher 357k nps, H=1024 student 232k nps (my student inference is a naive autovectorised
  implementation; the teacher's SFNNv18 code is heavily optimised), H=256 student 484k nps.
Conclusion: ~8 CPU-hours of distillation on 4 cores yields a net ~380 Elo below master at equal nodes.
It does not approach master strength; R1-R6 are about *relative* efficiency of data sources.

## E9: cheaper game-position baseline (correction to R1's ratio) + E8 vs E4
Game positions from the E3 1000-node self-play (seeds 3001-3004; the n1000_r02/r08 runs are identical
games, so one run's 749,661 game positions are used): 1,870 CPU-s, i.e. 4x cheaper per position than run1.
- game_n1000_p1.5e7 (vgame 0.02590, vtree 0.03457) vs game_f1 (run1 games, 7,490 CPU-s): **+69 +/- 44**
  (LOS 99.9%). Faster, weaker self-play gives *better* game-position data, consistent with R3: more
  imbalanced/varied positions.
- tree_f32 (234 CPU-s) vs game_n1000_p1.5e7 (1,870 CPU-s): **+61 +/- 45** (LOS 99.7%).
=> Against the best game-position pipeline tested, the measured saving is >= 8x generation compute with the
tree student still stronger; from the tree scaling slope (~+150 Elo per doubling at that end) equal
strength would be reached at ~175 CPU-s, i.e. ~10x. The earlier ">= 32x" was relative to the 5000-node
game baseline and overstates the advantage over a tuned game pipeline. E10 tests even cheaper games.
- E8 (H=1024, all 144M tree positions) vs E4 tree_h1024 (run1 only): +125 +/- 44 (200 games).

## E10: even cheaper game-position pipelines (second correction to R1's ratio)
Game positions from 300- and 100-node self-play, same 1,870 CPU-s budget (seeds 5001-5004):
1.42M and 2.08M game positions. H=256, teacher labels. 200-300 games each, 20k nodes:
| student | vgame | vtree | Elo |
|---|---|---|---|
| game_n300_p1.5e7 | 0.02516 | 0.03332 | vs game_n1000: +91 +/- 47 |
| game_n300_p3e7 | 0.02564 | 0.03343 | vs game_n1000: +115 +/- 47 |
| game_n100_p1.5e7 | 0.02466 | 0.03379 | vs game_n1000: +108 +/- 44 |
| game_n100_p3e7 | 0.02448 | 0.03269 | vs game_n1000: +151 +/- 50 |
Decisive direct match: tree_f32 (234 CPU-s) vs game_n100_p3e7 (1,870 CPU-s): **-78 +/- 37** (300 games).
Game-data quality rises steeply as self-play gets faster/weaker (5000 -> 1000 -> 100 nodes: 0, +69, ~+220
on the run1 scale), consistent with the mechanism (diverse, unbalanced positions matter).
Revised R1: versus the best game pipeline found (100-node self-play), tree distillation reaches equal
strength with ~5x less generation compute (interpolating the tree curve between 234 and 468 CPU-s:
~330-360 CPU-s vs 1,870), and is ~+250-300 Elo stronger at equal compute (tree at 1,870 CPU-s: +465,
or ~+516 with the E3 settings). The earlier ">= 32x" and ">= 8x" figures were relative to weaker game
baselines and are superseded.

## E11: game-pipeline plateau and capacity with good game data (correction to R3)
- 30-node self-play (1,870 CPU-s, 2.37M game positions; per-move overhead dominates): vtree 0.03255;
  vs 100-node data: +19 +/- 48 (n.s.). The game pipeline plateaus around 30-100 nodes/move.
- H=512 on the 100-node game data (2.08M positions): vtree 0.03220; vs H=256 on the same data: +56 +/- 43
  (LOS 99.4%). So the E4 finding "capacity does not help game data" holds for standard 5000-node game data
  (734k positions) but not for larger/more diverse fast-game data, where H=512 gains about as much as
  with tree data (+70). Revised R3: capacity scaling is limited by data diversity; tree data has the most.

## E12: tree data from fast self-play (does the game-data trend transfer?) - no
Same 1,870 CPU-s budget, 25% of evaluate() calls recorded (seeds 7001-7004), H=256, 1e8 positions:
| setting | tree positions | vgame | vtree | vs E3 best (20000 nodes, 8%) |
|---|---|---|---|---|
| 300 nodes, 25% | 35.3M | 0.02227 | 0.02355 | -44 +/- 41 |
| 1000 nodes, 25% | 45.7M | 0.02078 | 0.02210 | -24 +/- 39 |
| 20000 nodes, 8% (E3) | 28.7M | 0.02050 | 0.02016 | 0 |
Opposite to game data: tree positions from deeper per-move searches are worth more (shallow trees are
dominated by near-root, highly correlated nodes). The vtree proxy again orders the three correctly.
Best tree recipe found: >= 5000-20000 nodes/move, record 8%+ of evaluate() calls.

## E13: headline ratio with the best settings found on both sides
Tree: 20000 nodes/move, 8% recording (E3 best), 4 procs x 59 s = 234 CPU-s (2.92M positions) and x 117 s =
468 CPU-s (6.32M), H=256, 3.2e7 training positions. Game: 100-node self-play, 1,870 CPU-s (E10 best,
game_n100_p3e7). 300 games each:
- tree_c234 vs game_n100_p3e7: -13 +/- 36 (n.s.) -> equal strength at **8x less** generation compute.
- tree_c468 vs game_n100_p3e7: **+61 +/- 38** (LOS 99.94%) at 4x less compute.
Final R1: with tuned settings on both sides, tree distillation matches the best game-position pipeline
with ~8x less data-generation compute (the ~5x interpolation in E10 used the untuned 5000-node/2% curve).
Note: the vtree proxy under-rates the fast-game students (vtree 0.0327 but Elo ~ tree students at
vtree ~0.029): R2 is reliable within a data family, less so across very different position sources.

# Phase II (branch claude/phase2-search-tree-training)

## P2 setup: architecture-identical fine-tuning of the production net
Rationale: a full SFNNv18 retrain is impossible on 4 CPUs, but its 32 dense layer stacks (1.13M of
~113M parameters) can be fine-tuned on CPU with the feature transformer frozen. Any change is then
architecture-identical (same speed), so fixed-node Elo is equal-time Elo and a positive result would
be directly fishtest-testable.
Infrastructure (verified):
- `tools/p2/nnue_stacks.py`: reads/writes the stacks (last 32 x 35,208 bytes of the .nnue; all 32 stack
  hashes 0x63337116; read->write round trip byte-identical).
- fork patch 0005: `ftdump` (u8[1024] FT output + bucket + Stockfish raw eval per record), `label`
  now flags positions whose search best move is a capture/promotion (Rec.src bit 7).
- `tools/p2/finetune.py`: float layer-stack model with fake quantisation; reproduces Stockfish's raw
  eval from ftdump features within rounding (mean |diff| 0.50, max 0.9995 internal units, 3,000 positions).
  Targets: search scores mapped through the inverse of scale_evaluation (optimism 0; inversion exact to
  <= 1 unit), so the fine-tuned raw output is calibrated for Stockfish's own rescaling.
Data P2-D1: from run1 (master self-play, 5000 nodes/move, seeds 101-104): 300k game positions (G), 300k
tree positions (T); held-out from val games (seeds 901-904): 20k VG, 20k VT. All re-labelled by the
same 5000-node search (`label`), so labels are identical in kind for G and T.

## PRE-REGISTRATION P2-F1 (written before any fine-tuning result)
Question: fine-tuning master's layer stacks on search-labelled positions, do tree positions beat game
positions (same label procedure, same count, same hyperparameters)?
- Arms: FT-G (game), FT-T (tree), FT-M (50/50). Default: quiet filter (drop best-move-is-capture) applied
  identically to all arms, as nnue-pytorch's smart fen skipping does. Same steps, LR, anchor, seed.
- Hyperparameters (LR, anchor, steps) chosen ONLY on validation loss on the pooled VG+VT set, using FT-M,
  then frozen for all arms.
- Primary endpoint: direct match FT-T vs FT-G, 20k nodes, >= 1000 games. Secondary: each vs master.
- A "Stockfish improvement" claim requires: candidate vs master > 0 with LOS >= 99% on >= 4000 games at
  fixed nodes AND confirmation at STC (10+0.1) on an independent opening seed.

## BUG FOUND (affects Phase I search-label results): recorded search scores are centipawns
`gendata`/`label` stored `Score::InternalUnits`, which Stockfish fills with `UCIEngine::to_cp(v)` =
round(100 v / a(material)), a ~ 300-400. Teacher labels (raw net eval) are true internal units. Phase I
trained "score"/"blend" students with sigmoid(score/340), i.e. on targets compressed ~3.4x.
Impact on Phase I:
- UNAFFECTED: all teacher-label results (R1 incl. E9/E10/E13, R2, R3, R4, E7/E8).
- INVALID: absolute strength of all search-score-label students (E1 game_score/game_blend, gq_score,
  gq_blend; "teacher labels beat search labels +91"; "WDL blending hurts"; "-600..-1300 unfiltered").
  Their evals were ~3x too small for Stockfish's pruning margins. These claims are withdrawn.
- E6 (tree_score vs gq_score, +80) compared two equally mis-scaled arms: internally consistent but at a
  pathological scale; to be re-run with correct units before it can be claimed.
- Adjudication thresholds in gendata (|score| >= 2500, <= 15) were in cp: only game lengths affected.
Fix: `tools/p2/finetune.py: cp_to_internal()` (exact inverse of to_cp up to rounding). Check: slope of
search value on scaled static eval is 1.07 (VG) / 1.11 (VT) after conversion (0.32 before).

## P2-1: production master net vs its own 5000-node search (corrected units)
Static eval mapped through scale_evaluation vs search value, sigmoid-MSE (scale 340), held-out sets:
| set | n | best move capture | MSE all | MSE quiet | quiet & material-balanced | quiet & imbalanced | share |err|>300, quiet |
|---|---|---|---|---|---|---|---|
| VG game positions | 19,675 | 19% | 0.0144 | 0.0078 | 0.0086 | 0.0063 | 17.9% |
| VT tree positions | 19,283 | 36% | 0.0479 | 0.0141 | 0.0166 | 0.0112 | 30.3% |
The production net (trained on game positions of Leela data) is ~1.8x further from search on quiet tree
positions than on quiet game positions, also when material is balanced. Consistent with the
distribution-mismatch hypothesis, but tree positions may also be intrinsically harder: fine-tuning
(P2-F1) tests whether the gap is learnable.

## P2-F1 RESULT (pre-registered primary endpoint)
Arms: 186,365 quiet positions each, 5000-node search labels (corrected units), lr 1e-4 (selected on pooled
validation with the mixed arm), 1500 steps x 8192, seed 0, layer stacks only.
Validation MSE vs search (VG / VT): master 0.01263 / 0.0451; FT-G 0.01099 / 0.0402; FT-T 0.01388 / 0.0326;
FT-M 0.0116 / 0.0347. Each arm specialises to its own distribution.
- PRIMARY: FT-T vs FT-G = **-27.2 +/- 12.7** (1000 games, 20k nodes, LOS 0%). Hypothesis rejected in this setting.
- FT-T vs master = -66.8 +/- 13.4: fine-tuning on this label set degrades the production net.
Interpretation (to be tested): (1) 186k positions of 5000-node labels cannot improve a net trained on
billions of Leela positions; fine-tuning drifts. (2) For a strong evaluator, accuracy matters most at
decision-relevant nodes (PV / near the alpha-beta window); uniformly sampled tree positions are mostly
refuted subtrees, so fitting them trades away accuracy where it matters (FT-T's game-position MSE got worse).

## PRE-REGISTRATION P3 (before results): re-distillation of the production layer stacks
Frozen master feature transformer; layer stacks re-initialised at random and trained on master's own
eval (teacher labels, ftdump sfEval) - the production-architecture analogue of Phase I's R1.
Arms (600k positions each, all non-check positions, no quiet filter): RS-G (run1 game positions, 5000-node
self-play), RS-F (game positions from 100-node self-play = best game pipeline of Phase I), RS-T (run1 tree).
LR chosen on pooled VG+VT validation (teacher target) with a G+T mix arm, then frozen.
Primary endpoint: RS-T vs RS-F, 1000 games at 20k nodes. Secondary: RS-T vs RS-G, each vs master.
