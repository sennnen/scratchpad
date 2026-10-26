"""Record I/O for the 40-byte training records written by the research fork (see research.h)."""
import numpy as np
import torch

REC = np.dtype([("board", np.uint8, 32), ("stm", np.uint8), ("result", np.uint8),
                ("src", np.uint8), ("rule50", np.uint8), ("teacher", np.int16), ("score", np.int16)])
assert REC.itemsize == 40


def load(paths, src=None, max_n=None):
    arrs = []
    for p in paths:
        a = np.fromfile(p, dtype=REC)
        if src is not None:
            a = a[(a["src"] & 1) == src]  # bits 1-7 carry ply / label flags
        arrs.append(a)
    a = np.concatenate(arrs) if arrs else np.zeros(0, REC)
    if max_n is not None:
        a = a[:max_n]
    return a


def features(recs):
    """Return (idx_stm, idx_nstm, bucket) as torch tensors; idx padded with 768, shape (N,32)."""
    b = torch.from_numpy(np.ascontiguousarray(recs["board"]))  # N,32
    lo = (b & 15).to(torch.int64)
    hi = (b >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=2).reshape(-1, 64)  # square order a1..h8
    stm = torch.from_numpy(recs["stm"].astype(np.int64)).unsqueeze(1)  # N,1
    occ = codes != 0
    pt = (codes & 7) - 1
    col = codes >> 3
    sq = torch.arange(64, dtype=torch.int64).unsqueeze(0)
    rel_s = (col != stm).to(torch.int64)
    idx_s = (rel_s * 6 + pt) * 64 + (sq ^ (56 * stm))
    idx_n = ((1 - rel_s) * 6 + pt) * 64 + (sq ^ (56 * (1 - stm)))
    idx_s = torch.where(occ, idx_s, torch.full_like(idx_s, 768))
    idx_n = torch.where(occ, idx_n, torch.full_like(idx_n, 768))
    # compact to 32 slots: sort so padding goes last
    order = torch.argsort((~occ).to(torch.int8), dim=1, stable=True)[:, :32]
    idx_s = torch.gather(idx_s, 1, order)
    idx_n = torch.gather(idx_n, 1, order)
    npieces = occ.sum(1)
    bucket = torch.clamp((npieces - 2) // 4, 0, 7)
    return idx_s, idx_n, bucket
