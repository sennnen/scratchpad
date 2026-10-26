# Next steps (ranked by expected value per compute)

1. **Scale test of tree-position distillation at Stockfish-relevant size.** Distill the master net into a
   HalfKAv2_hm / L1>=512 student with nnue-pytorch on GPU, comparing (a) game positions vs (b) teacher
   search-tree positions at equal generation compute. R1 predicts a large gap; R2 says to validate on
   tree positions. This is the decisive experiment for upstream relevance and needs a GPU.
2. **Bootstrapping new architectures.** When an architecture changes, Stockfish retrains from scratch on
   Leela data. Initialising (or pre-training) the new net by tree distillation from the current master
   could cut the GPU budget of architecture experiments; test as: distill -> short fine-tune on Leela
   data vs full Leela training, at matched GPU hours.
3. **Augment Leela training data with master-labelled tree positions.** Upstream nets are trained only on
   game positions. R1/R3 suggest the eval is weakest exactly on positions search visits but games do not.
   Mixing a small fraction of current-master-labelled tree positions into Leela-data training is cheap
   and directly testable on fishtest.
4. **Use tree-position validation loss for net selection** (R2) instead of game-position loss.
5. On-policy (DAgger) data and generation settings: see E2/E3 results in RESEARCH_LOG.md.
6. Equal-time strength of distilled students with incremental int16 inference (patch 0002).
