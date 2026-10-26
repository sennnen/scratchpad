# Results (validated findings)

All numbers measured in this session on 4 vCPUs (no GPU). Elo values come from paired-opening matches
(UHO_4060_v4, disjoint from the training book) at a fixed 20,000 nodes/move, using a joint Bradley-Terry
fit with pentanomial bootstrap 95% intervals. Students: (768 -> H) x 2 perspectives, SCReLU,
8 material output buckets, running inside an otherwise unmodified Stockfish master search.
Teacher: Stockfish master net `nn-c9e480e92425` (SFNNv18).

## R1. Distilling from teacher search-tree positions is >= 32x more compute-efficient than from game positions

Every `evaluate()` call in a teacher search yields a (position, teacher eval) pair for free. Recording 2%
of them gives ~34x more labelled positions per generation CPU-second than the game positions of the same
self-play, *and* those positions follow the distribution the evaluator is queried on during search.

Elo vs generation compute (H=256; anchor: game-position student trained on all data = 0):

| generation compute | tree-position student | game-position student (teacher labels) |
|---|---|---|
| 234 CPU-s (1/32) | **+111** [75, 151] | |
| 468 CPU-s (1/16) | +264 [214, 311] | |
| 936 CPU-s (1/8) | +398 [353, 448] | |
| 1,872 CPU-s (1/4) | +465 [413, 516] | -362 [-430, -303] |
| 3,745 CPU-s (1/2) | +520 [466, 576] | -159 [-202, -113] |
| 7,490 CPU-s (1) | +545 [493, 615] | 0 |

Conventional labels on the same game positions (best variant: quiet positions only, 5000-node search
score): -113 [-165, -66]; with 0.7 score + 0.3 game result: -252 [-315, -198]; without the quiet filter:
-600 to -1300.

- Proven (at this scale): the tree student trained on 1/32 of the generation compute beats the
  game-position student trained on all of it (direct match +104 +/- 49, 200 games) and beats the best
  conventional-label student by ~+224 Elo. Counting training too, total CPU is ~300 s vs ~7,556 s (>=25x).
- Sample-matched (734k positions each): tree positions still +207 [162, 253] over game positions, so
  the gain is not only quantity; the position distribution itself matters.

## R2. Search-tree validation loss predicts strength; game-position validation loss does not

Across 11 teacher-label students: Pearson(Elo, -MSE_tree) = 0.996, Spearman = 1.000;
Pearson(Elo, -MSE_game) = 0.850, Spearman = 0.800. The game-position student has *lower* game-position
validation loss (0.0249) than three tree students that beat it by 111-264 Elo.

## R3. Mechanism evidence

Weaker students searched much shallower at equal nodes (depth 8-9 vs 12-16 in sampled games) and hung
material: eval errors on off-distribution (material-imbalanced, tactical) positions corrupt pruning
decisions, not just leaf scores. Tree positions cover exactly those positions.

## Limitations (apply to all of the above)
- Small students (768->256) far weaker than the teacher; transfer to Stockfish-size nets is untested.
- Fixed-node matches (equal nodes, not equal time); all students share one architecture and speed.
- One teacher, one generation setting (5000 nodes/move, 2% recording) for R1; 200 games per pair.
- Distillation presupposes a strong teacher network; this is a method for transferring/bootstrapping
  strength (e.g. into new architectures), not for learning chess from scratch.

(Pending: E2 on-policy/DAgger, E3 generation settings, E1c replication, E4 capacity scaling.)
