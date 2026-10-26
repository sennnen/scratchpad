# Next steps (ranked by expected value per compute)

1. **Full-scale test of tree-position distillation (needs a GPU).** Distill the current master into a
   Stockfish-architecture student with nnue-pytorch, comparing game positions vs teacher search-tree
   positions (the `gendata ... treerate` patch produces both) at equal generation compute, and validate on
   tree positions (R2). R1 and R3 predict that the gap *grows* with capacity, so this is the decisive
   experiment for upstream relevance. Also run the on-policy step (R4): self-play with the student,
   teacher-label its tree positions, retrain.
2. **Augment Leela-data training with master-labelled tree positions.** Upstream nets see only game
   positions. R3 and R5 suggest evaluators are weakest exactly on the off-game positions that search
   visits. Mixing a small fraction of current-master-labelled tree positions into Leela-data training is
   cheap to try and directly testable on fishtest.
3. **Bootstrapping new architectures.** Architecture changes currently need full retraining on Leela
   data. Pre-train by tree distillation from the current master, then fine-tune on Leela data, and compare
   with full training at matched GPU hours.
4. **Net selection by tree-position validation loss** (R2) for nets trained on the same targets.
5. **Cheaper teacher-free pipelines (R5).** Self-generated data for engines without a teacher: sample
   positions from the engine's own search trees and label them by search; tune the label search for
   non-quiet positions (e.g. qsearch-resolved targets).
6. **More DAgger iterations and larger students.** Only one on-policy iteration at H=256 was tested.
7. **Faster student inference.** The H=1024 student runs at ~0.65x master's nps (naive autovectorised
   code). Equal-time strength needs SIMD-tuned int16/int8 inference (Stockfish-style).
8. **Game-pipeline tuning.** Faster self-play gave better game-position data (E9/E10); a fuller sweep
   (randomised openings, nodes, filtering) would tighten the R1 ratio against the best possible game pipeline.
