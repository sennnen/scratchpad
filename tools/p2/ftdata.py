"""FtRec (1068 bytes) I/O: 40-byte Rec + int16 sfEval + u8 bucket + u8 pad + u8 ft[1024]."""
import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data import REC
FTREC = np.dtype([('rec', REC), ('sfeval', '<i2'), ('bucket', 'u1'), ('pad', 'u1'), ('ft', 'u1', 1024)])
assert FTREC.itemsize == 1068

def load(path, mmap=True):
    return np.memmap(path, dtype=FTREC, mode='r') if mmap else np.fromfile(path, dtype=FTREC)
