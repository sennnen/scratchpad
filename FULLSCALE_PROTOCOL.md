# Full-scale protocol: search-tree positions in Stockfish NNUE training

Status: **ready to execute but not run.** It needs GPU training with nnue-pytorch and fishtest
evaluation, neither of which is available in the CPU-only environment used for this research. The
expected effect size is uncertain (see "Prior").

## Question
Does adding positions sampled from search trees (`evaluate()` calls of a strong engine), labelled by the
current master net, to the standard Leela-data training recipe produce a stronger Stockfish network?

## Prior from this study (CPU, small students + production layer stacks)
- From-scratch students (features learned): very large gains from tree positions (R1, R3, P4, B4:
  +531 Elo from tree-trained features at equal readout).
- Production architecture with master's frozen feature transformer: small gain (readout only;
  +14 +/- 11 pooled at equal count; +43 vs fast-game positions).
- B4: the readout effect is small whenever the features already cover the query distribution.
  Master's FT was trained on billions of positions, so the full-scale gain could be small. The open question
  is whether FT *training* still benefits at that scale. Master's static eval is 1.8x further from its own
  search on quiet tree positions than on quiet game positions (P2-1), so there is measurable room.
- Fine-tuning master on small search-labelled sets hurts (P2-F1). Run full trainings, not fine-tunes.

## Data generation (CPU workers, e.g. fishtest-style)
1. Stockfish master at the training base revision plus `sf-patches/0001` (gendata + recorder),
   `0006` (ply metadata) and optionally `0007` (search context).
2. Per worker: `gendata nodes 20000 games N out tree_<seed>.bin book UHO_Lichess_4852_v1.epd seed <s> treerate 0.08`
   (E3/E12: deeper searches and a higher recording rate give better tree data; avoid fast-game trees).
   Each 20k-node search yields ~1,600 recorded positions at 8%. The teacher label is the master raw eval
   (`Rec.teacher`, internal units), produced as a free byproduct.
3. Sample uniformly from the recorded calls. Do **not** select subsets (B1/B2/B3: every selection rule
   tested, near-window, by ply, main search only, or by student error, was worse than uniform).

## Conversion to nnue-pytorch input (to be implemented; spec)
`Rec` (40 bytes, `src/research.h`) to a binpack `TrainingDataEntry`:
- `pos`: board + side to move (castling/en-passant are not stored: use none);
- `score`: `Rec.teacher` (master raw eval, i.e. a distillation target in the net's own output units);
- `move`: a legal non-capture if one exists (the move is only used by skipping filters);
- `ply`: game ply of the search root plus `ply-from-root` (patch 0006);
- `result`: unknown, so use a separate dataset stream with `wld_filtered=False` and `filtered=False`.
  P7/P8 found that dropping capture positions hurts with teacher labels: -22 and -48 Elo for game data.

## Arms (equal GPU-hours and steps; pre-register before running)
- A, reference: the current master training recipe (nnue-pytorch scripts / fishtest net description).
- B5 / B20: recipe A with 5% / 20% of each batch drawn from the tree-position stream.
- C (control for label quality): recipe A with 5% / 20% of master-labelled *game* positions from the same
  self-play games. This separates "teacher labels" from "tree positions" (the factorial of P4).
- D (optional, after B wins): one on-policy iteration. Generate tree positions with the B-net as
  evaluator, label them with master, and retrain. Expect a small effect (P5: +6 +/- 13 for a near-teacher student).

## Evaluation
fishtest SPRT on each arm vs A: STC 10+0.1 [0, 2], then LTC 60+0.6 [0.5, 2.5]. Claim an improvement only
if B passes both **and** beats C (the tree effect, not the teacher-label effect).

## Cost estimate
Generation: E3 measured ~15,000 recorded tree positions per CPU-second (20k nodes/move, 8% recording,
master, 1 thread on this machine), so 1e9 positions cost ~6.7e4 CPU-s (~18 core-hours), negligible next
to GPU training. Each recorded position carries its teacher label for free. Conversion is I/O bound.
Training uses the standard recipe's GPU budget per arm.
