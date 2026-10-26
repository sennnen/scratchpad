# Results

# Phase II: does training on search-tree positions transfer to production Stockfish?

**Short answer.** Partly. The effect is real and mechanistically understood, but in the production
architecture (with Stockfish's own frozen feature transformer) it shrinks to a small gain, and no
CPU-feasible modification produced a stronger Stockfish. The decisive full-scale test (training the
feature transformer itself) is specified in FULLSCALE_PROTOCOL.md.

**Setup.** Production net `nn-c9e480e92425` (SFNNv18, master `ff3a6c50`). Its 110 MB feature transformer
is frozen; its 32 dense layer stacks (1.13M parameters) are fine-tuned or re-trained on CPU with an exact
float model and fake quantisation (exported nets load in Stockfish and reproduce the trainer's evals within
rounding). Identical architecture means identical speed, so fixed-node Elo equals equal-time Elo.
Matches: 20k nodes/move, UHO_4060_v4 (disjoint from training openings), paired; key comparisons were
pre-registered in RESEARCH_LOG before results.

## Confirmed (pre-registered or replicated, LOS >= 99%)

| # | Finding | Numbers |
|---|---|---|
| Q1 | The production net's static eval is ~1.8x further from its own 5000-node search on quiet tree positions than on quiet game positions, also when material is balanced. | MSE 0.0141 vs 0.0078 (40k held-out positions) |
| Q2 | Fine-tuning master's stacks on small search-labelled sets degrades it, monotonically in the tree fraction. | game -25, mix -40, tree -67 (+/-13, 1000 games each); tree vs game -27 +/- 13 |
| Q3 | Re-distilling the stacks from master's evals: tree positions beat fast-game positions. | +43.0 +/- 13.7 (1000 games) |
| Q4 | 2x2 factorial (identical positions, re-initialised stacks): tree beats game under **both** label types. | teacher labels +82.8 +/- 16.6; search labels +43.7 +/- 18.7 |
| Q5 | The quiet filter (drop capture-best positions) helps tree data and **hurts game data** under teacher labels. | quiet game -146 vs unfiltered -98 (vs master); SEE-filtered game -22 +/- 17 |
| Q6 | Most of the tree advantage lives in **feature learning**; the readout benefits only when features lack coverage. | tree-trained vs game-trained features +531 +/- 104; readout effect +186 (game features) vs +3 (tree features) |
| Q7 | Selecting "informative" tree positions loses to uniform sampling: near the alpha-beta window -531, far -119, main search only -80, by student error -100 to -849. | B1/B3/B2, 200 games each |
| Q8 | On-policy data gives weak students a persistent edge over matched off-policy data, but iterating does not keep improving a capacity-limited student; no detectable benefit for a near-teacher student. | pooled over 3 iterations +17.8 [9.1, 26.6] (3,800 games); per iteration +22.6 / +3.1 / +24.0; production +6 +/- 13 |

## Measured but small or inconclusive
- Tree vs standard 5000-node game positions, unfiltered teacher labels, production stacks: +12.9 +/- 13.3
  (300k) and +15.1 +/- 17.6 (600k); pooled about +14 +/- 11. P6 data scaling: no clear data multiplier
  (T - G of +9, -11, +15, +1 at 75k-600k, +/-22 each).
- SEE-filtered tree positions vs game positions: +3.8 +/- 13.7 (the cheap static filter does not
  reproduce the search-quiet effect).
- Coverage: KL(query distribution || training set) 0.013 (tree) < 0.184 (game) < 0.547 (fast game),
  the same order as P3 Elo (-43, -58, -81 vs master).

## Not achieved
- **No stronger Stockfish.** Every CPU fine-tune of the production net was weaker than master (Q2), and
  re-distilled stacks reach at best -43 vs master. No candidate merits fishtest submission.
- **No order-of-magnitude saving at production scale.** The >= 8x saving holds for small students that
  learn their own features; for the production readout it is ~1x to 1.6x.

## Mechanism (synthesis)
An evaluator inside alpha-beta search is queried on the positions search visits, which are more unbalanced,
more tactical and further from good play than game positions. When an evaluator must *learn features*
(small students, or any from-scratch training), covering that query distribution is worth hundreds of Elo
at equal data (Phase I R1, B4), and every attempt to reweight towards "decision-relevant" or "hard" subsets
damaged coverage and lost. When features are already good (Stockfish's feature transformer trained on
billions of positions), the remaining readout is largely distribution-insensitive, so the gain shrinks to
~+14 Elo. Fine-tuning a strong net on a small, shallow-search-labelled sample moves it away from an
optimum it reached with far better data, and tree positions, whose search labels are noisiest, move it
furthest. On-policy data corrects the covariate shift between student and teacher trees: a persistent ~+18 Elo
for a weak student, nothing detectable once the student is within ~40 Elo of its teacher.

## Bugs found and corrected in Phase II
- Recorded search scores were centipawns, not internal units: all Phase I search-label results were
  withdrawn, and teacher-label results are unaffected.
- The joint-Elo fitter could stop at a non-optimal point on star-shaped match graphs: fixed (exact
  Newton). All earlier ladders reproduce exactly; only one Phase II ladder file was affected.

# Phase I (small students)

> **Phase II correction (2026-10-26):** recorded search scores were centipawns, not internal units
> (see RESEARCH_LOG "BUG FOUND"). All results trained on *teacher* labels (R1-R4, R6) are unaffected.
> All absolute numbers for *search-score-label* students are withdrawn (the "conventional labels"
> bullets in R1, the WDL-blending note), and R5 needs a re-run with correct units.

All numbers were measured in this session on a 4-vCPU cloud container (AVX-512 VNNI, 15 GB RAM, no GPU).

**Setup.** Teacher: Stockfish master `ff3a6c50` with net `nn-c9e480e92425` (SFNNv18). Students: small
NNUE-style nets ((768 or 8x768 king-bucketed) -> H) x 2 perspectives, SCReLU, 8 material buckets, plugged
into the *unmodified* Stockfish search via a fork option (`StudentFile`). Data: self-play of the teacher
from UHO_Lichess_4852_v1 openings. Matches: paired openings from a *different* book (UHO_4060_v4), fixed
20,000 nodes/move unless stated, joint Bradley-Terry fits with pentanomial-bootstrap 95% intervals.
"CPU-s" = CPU-seconds of data generation (the dominant cost; training is listed separately).

## Summary

| # | Finding | Status |
|---|---|---|
| R1 | Distilling from positions sampled out of the teacher's **search trees** (its `evaluate()` calls) instead of game positions needs **~8x less generation compute for equal strength** than the best game-position pipeline found (100-node self-play), with tuned settings on both sides (E13), and is ~+250-300 Elo stronger at equal compute. Against standard 5000-node self-play data the saving is >= 32x. | Proven at small scale; replicated; robust over 5k-80k nodes |
| R2 | Validation loss on held-out **search-tree** positions predicts Elo almost perfectly (Pearson 0.995, 18 students); loss on game positions does not (0.92, misranks by up to 264 Elo). | Proven within one teacher, same label type |
| R3 | Capacity scaling is limited by data diversity: on standard 5000-node game data, more width or king buckets buy nothing or hurt; on tree data every width doubling adds ~+70 Elo; on fast-game (100-node) data H=512 adds +56. | Proven at H <= 1024 (game side to H=512) |
| R4 | On-policy (DAgger-style) data: teacher-labelled positions from the **student's own** search trees beat the same amount of teacher-tree data by +22.6 [9.6, 35.7] (1,800 games). | Proven for one iteration at H=256 |
| R5 | The position effect is not specific to distillation: with conventional 5000-node search-score labels on both sides, tree positions beat quiet game positions by +80 +/- 36 at equal label count and cost. | Single experiment (300 games) |
| R6 | ~10 CPU-hours of distillation yields a student ~380 Elo below master at equal nodes (worth master at ~1/5 of the nodes) and -330 at equal time. It does **not** approach Stockfish strength. | Measured; negative for "comparable strength" |

## R1. Search-tree positions vs game positions (distillation, H=256)

Every `evaluate()` call in a teacher search is a (position, teacher eval) pair at no extra cost. Recording
2% of them yields ~34x more labelled positions per CPU-s than the game positions of the same self-play, and
those positions follow the distribution an evaluator is queried on during search.

Elo vs generation compute (anchor: game-position student on all of run1 = 0; game rows at 1,870 CPU-s
chained through direct matches):

| generation compute | tree-position student | game-position student |
|---|---|---|
| 234 CPU-s | **+111** [75, 151] | |
| 468 CPU-s | +264 [214, 311] | |
| 936 CPU-s | +398 [353, 448] | |
| 1,870 CPU-s | +465 [413, 516] (E3 settings: ~+516) | 5000-node games -362; 1000-node +69; 300-node ~+184; **100-node ~+220** |
| 3,745 CPU-s | +520 [466, 576] | 5000-node games -159 |
| 7,490 CPU-s | +545 [493, 615] | 5000-node games 0 |

- Decisive direct matches against the best game student (100-node self-play, 1,870 CPU-s), with the best
  tree settings (20k nodes/move, 8% recording, E13): tree at 234 CPU-s **-13 +/- 36** (equal strength at
  **8x less compute**); tree at 468 CPU-s **+61 +/- 38** (LOS 99.94%). (With the untuned 5000-node/2%
  tree setting, 234 CPU-s gave -78 +/- 37, i.e. ~5x.) At equal compute (1,870 CPU-s) the tree student is
  ~+250-300 Elo stronger.
- Against the standard 5000-node game baseline: 234-CPU-s tree student +104 +/- 49 vs 7,490 CPU-s of
  game data (>= 32x), replicated on three disjoint data subsets (+147, +133, +81, each +/- ~47, LOS >=
  99.99%) and at 5k/20k/80k nodes (+123/+104/+141).
- How the ratio evolved: my first baselines (5000-node games) gave ">= 32x"; cheaper game generation
  (E9, E10) improved the game pipeline by ~+220 Elo and reduced the ratio; tuning the tree side (E3, E13) brought it back to ~8x. 30-node games did not
  improve further (+19 +/- 48 vs 100-node, E11), so the game pipeline appears to plateau there.
- Sample-matched (734k positions each, same 7,490 CPU-s run): tree +207 [162, 253] over game positions, so
  the distribution matters, not only the quantity.
- Generation settings (equal compute): recording 8% instead of 2% and/or 20k instead of 5k nodes/move
  adds up to +51 +/- 38 for tree data. Unlike game data, tree data gets *worse* from fast self-play
  (300 nodes/25%: -44 +/- 41; 1000 nodes/25%: -24 +/- 39 vs 20k nodes/8%, E12).
- Conventional labels on game positions are worse still: quiet positions + 5000-node search score -113
  [-165, -66]; with 0.7 score + 0.3 WDL -252; unfiltered search labels -600 to -1300.

## R2. Which validation loss to trust

18 teacher-label students (H 256-1024, two feature sets, game and tree data):
Pearson(Elo, -MSE_tree) = 0.995, Spearman 0.988; Pearson(Elo, -MSE_game) = 0.920, Spearman 0.882.
The game-position student has lower game-position loss (0.0249) than three tree students that beat it by
111-264 Elo. The proxy also ranked all E3 and E12 generation settings correctly. It is less reliable across very
different position sources: fast-game students score worse vtree than tree students of equal Elo (E13). It does **not** transfer
across label types (search-labelled students have worse teacher-MSE but can be stronger, E6).

## R3. Capacity scaling depends on data diversity

Same run1 data (7,490 CPU-s), joint ladder (anchor game H=256 = 0), direct match in parentheses:

| student | tree data | game data |
|---|---|---|
| H=256 | +545 | 0 |
| H=512 | +613 (+70 +/- 38) | +2 (+2 +/- 39) |
| H=1024 | **+683** (+72 +/- 39) | -5 (-7 +/- 46) |
| 8 king buckets, H=256 | +605 (+58 +/- 39 vs H=256) | -133 (-133 +/- 44) |
| 8 king buckets, H=512 | +654 | |

Mechanism evidence: tree positions are more unbalanced than game positions (material imbalance >= 3 pawns:
32% vs 24%; median |teacher eval| 506 vs 317 internal units), and weak students search shallower at
equal nodes (depth 8-9 vs 12-16) and hang material: eval errors off the game distribution corrupt pruning.
At full data the gap grows with search effort (+545 at 20k, +655 at 80k nodes). Consistently, faster and
weaker self-play produced much better game-position data (1000 nodes +69, 100 nodes ~+220 vs 5000-node
games at <= 1/4 of the compute; plateau at 30-100 nodes), and on that larger, more diverse game data
capacity helps again (H=512 vs H=256: +56 +/- 43, E11). The table above uses 5000-node game data.

## R4. On-policy data (one DAgger iteration)

Student S0 (H=256, tree data) plays 6,000 self-play games; 2% of its evaluate() calls are labelled by
the teacher (24.1M positions, ~7,730 CPU-s). Compared with 25.3M more teacher-tree positions (~7,580 CPU-s),
added to the same 25M base, H=256, 2e8 training positions: head-to-head **+22.6 [9.6, 35.7]** (1,800
games); vs base: on-policy +36.5 [11.7, 58.9], more teacher data +12.3 [-11.3, 36.0].

## R5. Without a teacher: search-labelled tree positions

600k random tree positions re-labelled by 5000-node searches vs 594k quiet game positions with 5000-node
in-game search labels (same label type, count and per-label cost): **+80 +/- 36** (300 games, LOS 100%).

## R6. Escalation and the gap to master

H=1024 on all ~144M tree positions of the session (~28,400 generation CPU-s + 7,625 training CPU-s,
~10 CPU-hours total): +125 +/- 44 over H=1024 on run1 only. Versus master (same search):
-382 +/- 80 at equal nodes; -56 +/- 45 when master gets 1/4 of the nodes; +409 +/- 82 at 1/16; -330 +/- 54
at 10+0.1 (my student inference runs at ~0.65x master's nps; master's is far more optimised).
The int16 incremental student path is strength-neutral vs float (+9 +/- 29, 300 games).

## Negative and engineering findings
- PyTorch CPU (EmbeddingBag) trained this net at ~12k pos/s; a custom C++ trainer reached ~770k pos/s on
  4 threads (H=256), ~195k pos/s at H=1024.
- More teacher-tree data saturates at H=256 (+12, n.s.) although validation loss keeps improving.
- WDL blending with 5000-node self-play results hurt (-139 vs pure search score) at this data scale.
- A float full-refresh student is not faster than master's incremental SFNNv18; speed needs incremental
  int16 accumulators (implemented, board-diff based, exact within rounding).

## Limitations
- Students are small and remain far below master; whether R1-R5 hold for Stockfish-size nets trained on
  GPUs is the key open question (NEXT_STEPS #1).
- Most comparisons are 200-300 games at fixed nodes; errors are +/-35-50 Elo per pair. Effects claimed as
  proven have LOS >= 99.7% and replication or monotone trends.
- One teacher, one opening-book family; the search over game-generation settings was limited.
- Distillation presupposes a strong network. R1-R4 describe how to transfer strength cheaply (e.g. into new
  architectures), not how to learn chess from scratch. R5 is the teacher-free variant.

## Tier assessment
Tier B (valuable original research): a method with large, replicated, controlled gains in sample and
compute efficiency at small scale, a mechanism, and evidence that the gain grows with model capacity and
search effort. Not Tier A/S: no full-scale NNUE, no improvement over Stockfish itself.
