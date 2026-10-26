import struct
import numpy as np
import torch
import torch.nn as nn

OUT_SCALE = 340.0  # internal units per unit of network output (sigmoid scale)


class Student(nn.Module):
    """(768 -> H) x 2 perspectives, SCReLU, -> NB output buckets. Matches research.cpp."""

    def __init__(self, H=256, NB=8):
        super().__init__()
        self.H, self.NB = H, NB
        self.ft = nn.EmbeddingBag(769, H, mode="sum", padding_idx=768)
        self.ftb = nn.Parameter(torch.zeros(H))
        self.out = nn.Parameter(torch.zeros(NB, 2 * H))
        self.outb = nn.Parameter(torch.zeros(NB))
        with torch.no_grad():
            self.ft.weight.normal_(0, 0.05)
            self.ft.weight[768].zero_()
            self.out.normal_(0, 1.0 / (2 * H) ** 0.5)

    def forward(self, idx_s, idx_n, bucket):
        a = self.ft(idx_s) + self.ftb
        b = self.ft(idx_n) + self.ftb
        x = torch.cat([a.clamp(0, 1) ** 2, b.clamp(0, 1) ** 2], dim=1)
        w = self.out[bucket]
        return (x * w).sum(1) + self.outb[bucket]

    def export(self, path):
        with open(path, "wb") as f:
            f.write(b"STUDNT01")
            f.write(struct.pack("<iif", self.H, self.NB, OUT_SCALE))
            f.write(self.ft.weight[:768].detach().numpy().astype("<f4").tobytes())
            f.write(self.ftb.detach().numpy().astype("<f4").tobytes())
            f.write(self.out.detach().numpy().astype("<f4").tobytes())
            f.write(self.outb.detach().numpy().astype("<f4").tobytes())
