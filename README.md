# Project: Break Stockfish (CPU-only research lab)

Autonomous research on cheaper NNUE training and stronger search, run entirely on a
4-core CPU cloud container (no GPU). See:

- `RESEARCH_LOG.md`: chronological log of hypotheses, experiments, decisions.
- `RESULTS.md`: validated findings with uncertainty.
- `EXPERIMENTS.csv`: structured record of every experiment.
- `REPRODUCE.md`: exact pinned setup and commands.
- `NEXT_STEPS.md`: strongest remaining directions.

Layout:
- `sf-patches/`: patches against pinned Stockfish master (data generation, student eval hooks).
- `tools/trainer/train.cpp`: fast multithreaded CPU trainer for small NNUE students.
- `tools/*.py`: record I/O, consistency checks, match wrapper (fastchess).
- `experiments/`: per-experiment metrics JSON and match results.
