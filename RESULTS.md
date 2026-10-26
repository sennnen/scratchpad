# Results

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
| R1 | Distilling from positions sampled out of the teacher's **search trees** (its `evaluate()` calls) instead of game positions gives far more strength per CPU-second: a 234 CPU-s tree student beats the best game-position students trained with 8x (1,870 CPU-s) and 32x (7,490 CPU-s) more compute. | Proven at small scale, replicated 4x, robust over 5k-80k nodes |
| R2 | Validation loss on held-out **search-tree** positions predicts Elo almost perfectly (Pearson 0.995, 18 students); loss on game positions does not (0.92, misranks by up to 264 Elo). | Proven within one teacher, same label type |
| R3 | With game positions, extra capacity or richer features buy nothing or hurt; with tree positions every doubling of width adds ~+70 Elo. The gap between data sources widens with model size (~+680 Elo at H=1024). | Proven at H <= 1024 |
| R4 | On-policy (DAgger-style) data: teacher-labelled positions from the **student's own** search trees beat the same amount of teacher-tree data by +22.6 [9.6, 35.7] (1,800 games). | Proven for one iteration at H=256 |
| R5 | The position effect is not specific to distillation: with conventional 5000-node search-score labels on both sides, tree positions beat quiet game positions by +80 +/- 36 at equal label count and cost. | Single experiment (300 games) |
| R6 | ~10 CPU-hours of distillation yields a student ~380 Elo below master at equal nodes (worth master at ~1/5 of the nodes) and -330 at equal time. It does **not** approach Stockfish strength. | Measured; negative for "comparable strength" |

## R1. Search-tree positions vs game positions (distillation, H=256)

Every `evaluate()` call in a teacher search is a (position, teacher eval) pair at no extra cost. Recording
2% of them yields ~34x more labelled positions per CPU-s than the game positions of the same self-play, and
those positions follow the distribution an evaluator is queried on during search.

Elo vs generation compute (anchor: game-position student on all of run1 = 0):

| generation compute | tree-position student | game-position student |
|---|---|---|
| 234 CPU-s | **+111** [75, 151] | |
| 468 CPU-s | +264 [214, 311] | |
| 936 CPU-s | +398 [353, 448] | |
| 1,870 CPU-s | +465 [413, 516] | -362 [-430, -303] (5000-node games); **+69 +/- 44 (1000-node games)** |
| 3,745 CPU-s | +520 [466, 576] | -159 [-202, -113] |
| 7,490 CPU-s | +545 [493, 615] | 0 (5000-node games) |

- Direct matches: 234-CPU-s tree student vs 7,490-CPU-s game student +104 +/- 49; replicated on three
  disjoint data subsets (+147, +133, +81, each +/- ~47, LOS >= 99.99%); at 5k/20k/80k nodes: +123/+104/+141.
- Against the best game pipeline tested (1000-node self-play, 1,870 CPU-s): the 234-CPU-s tree student
  still wins +61 +/- 45 (LOS 99.7%). **Measured saving: >= 8x generation compute at higher strength**
  (~10x at equal strength by interpolation); >= 32x versus the 5000-node game baseline. E10_PLACEHOLDER
- Sample-matched (734k positions each, same 7,490 CPU-s run): tree +207 [162, 253] over game positions, so
  the distribution matters, not only the quantity.
- Generation settings (equal compute): recording 8% instead of 2% and/or 20k instead of 5k nodes/move
  adds up to +51 +/- 38 for tree data, so the tree side of the ratio is also not at its optimum.
- Conventional labels on game positions are worse still: quiet positions + 5000-node search score -113
  [-165, -66]; with 0.7 score + 0.3 WDL -252; unfiltered search labels -600 to -1300.

## R2. Which validation loss to trust

18 teacher-label students (H 256-1024, two feature sets, game and tree data):
Pearson(Elo, -MSE_tree) = 0.995, Spearman 0.988; Pearson(Elo, -MSE_game) = 0.920, Spearman 0.882.
The game-position student has lower game-position loss (0.0249) than three tree students that beat it by
111-264 Elo. The proxy also ranked all five E3 generation settings correctly. It does **not** transfer
across label types (search-labelled students have worse teacher-MSE but can be stronger, E6).

## R3. Capacity only helps with tree data

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
weaker self-play (1000 nodes/move) produced *better* game-position data than 5000-node self-play (E9).

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
