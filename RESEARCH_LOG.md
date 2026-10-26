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
