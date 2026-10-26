# Next steps: ranked hypotheses after Phase II

Ranking = (probability the hypothesis is true and matters at full scale) x (value to Stockfish) / cost.

1. **Feature-transformer training with tree positions (GPU; FULLSCALE_PROTOCOL.md).** Phase II located the
   effect: it lives mainly in *feature learning* (B4: +531 from tree-trained features vs +3 from the readout
   on good features). Production nets were never trained this way. Master's static eval is 1.8x further from
   its own search on quiet tree positions than on game positions (P2-1), so there is measurable headroom.
   Prior for a fishtest-passing gain: modest (perhaps 20-30%), because billions of Leela positions may
   already give adequate coverage. Cost: ~18 core-hours of generation plus standard GPU training per arm.
2. **Re-examine smart fen skipping for distillation-style targets.** Under teacher labels, dropping capture
   positions from game data hurt in both tests (P7 -48, P8 -22). The default filter is designed for noisy
   search labels and may be suboptimal when labels are consistent static targets (e.g. when mixing in
   master-labelled data). Cheap to test in nnue-pytorch (`filtered=False` on a teacher-labelled stream).
3. **Cheap architecture screening by tree distillation.** On CPU, layer stacks re-learned from master's
   evals on 600k tree positions reach -43 Elo of master in ~8 minutes. Ranking candidate layer-stack
   architectures by tree distillation could pre-screen changes before GPU training. Untested: whether
   distillation rankings predict full-training rankings.
4. **On-policy (DAgger) data for weak students.** Gains scale with the student-teacher gap
   (+23 at ~380 Elo below the teacher, +6 +/- 13 at ~43 below). Useful when bootstrapping small or new
   architectures from a strong teacher, not for refining near-teacher nets. C1 (iterations 2-3) results are
   in RESEARCH_LOG.
5. **Teacher-free variant.** Search-labelled tree positions beat quiet game positions (P4: +44 +/- 19 in
   the production architecture, E6 earlier). Relevant for engines without a strong teacher.
6. Deprioritised after negative results: selecting "informative" positions by alpha-beta margin, ply,
   main-search vs qsearch, or student error (B1-B3, B2: all worse than uniform, some catastrophically);
   fine-tuning a strong net on small search-labelled sets (P2-F1); static SEE filtering as a cheap
   quiet proxy (P8).
