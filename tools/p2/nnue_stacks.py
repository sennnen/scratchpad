"""Read/write the 32 dense layer stacks of an SFNNv18 .nnue file (Stockfish master ff3a6c50).
File layout (logical order, little endian): header, feature transformer, then 32 x
  [u32 hash][fc_0: i32 bias[32], i8 W[32][1024]][fc_1: i32 bias[32], i8 W[32][64]][fc_2: i32 bias[1], i8 W[1][128]]
The layer stacks are the last 32*35208 bytes. Float semantics (verified against Stockfish, see verify):
  x  = ft_u8 / 128                       (1024)
  h0 = W0 x + b0,  W0 = W0q/128, b0 = b0q/16384          (32)
  a0 = [min(127, floor(128*h0^2))/128 (sqr), clamp(floor(128*h0),0,127)/128] -> 64 (u8 quantized)
  h1 = W1 a0 + b1, W1 = W1q/64, b1 = b1q/8192            (32)
  a1 = same pair activation on h1 -> 64
  o  = W2 [a0, a1] + b2, W2 = W2q/128, b2 = b2q/16384; out = o + (h0[30] - h0[31])
  eval (internal units) = trunc(out*16384 * 9600 / 16384) // 16  ~= 600 * out
"""
import numpy as np

NSTACK, L1, L2, L3 = 32, 1024, 32, 32
PER = 4 + L2 * 4 + L2 * L1 + L3 * 4 + L3 * 64 + 4 + 128
assert PER == 35208


def read_stacks(path):
    raw = open(path, 'rb').read()
    base = len(raw) - NSTACK * PER
    st = []
    for s in range(NSTACK):
        o = base + s * PER
        h = np.frombuffer(raw, '<u4', 1, o)[0]; o += 4
        b0 = np.frombuffer(raw, '<i4', L2, o).copy(); o += L2 * 4
        W0 = np.frombuffer(raw, 'i1', L2 * L1, o).reshape(L2, L1).copy(); o += L2 * L1
        b1 = np.frombuffer(raw, '<i4', L3, o).copy(); o += L3 * 4
        W1 = np.frombuffer(raw, 'i1', L3 * 64, o).reshape(L3, 64).copy(); o += L3 * 64
        b2 = np.frombuffer(raw, '<i4', 1, o).copy(); o += 4
        W2 = np.frombuffer(raw, 'i1', 128, o).reshape(1, 128).copy(); o += 128
        st.append(dict(hash=h, b0=b0, W0=W0, b1=b1, W1=W1, b2=b2, W2=W2))
    return raw[:base], st


def write_stacks(path, prefix, st):
    out = bytearray(prefix)
    for d in st:
        out += np.uint32(d['hash']).tobytes()
        for k, dt in (('b0', '<i4'), ('W0', 'i1'), ('b1', '<i4'), ('W1', 'i1'), ('b2', '<i4'), ('W2', 'i1')):
            out += np.ascontiguousarray(d[k]).astype(dt).tobytes()
    open(path, 'wb').write(bytes(out))


def to_float(d):
    return dict(W0=d['W0'] / 128.0, b0=d['b0'] / 16384.0, W1=d['W1'] / 64.0, b1=d['b1'] / 8192.0,
                W2=d['W2'] / 128.0, b2=d['b2'] / 16384.0)


def to_quant(f, hash_):
    def q(x, s, lo, hi):
        r = np.round(x * s)
        return np.clip(r, lo, hi)
    return dict(hash=hash_, W0=q(f['W0'], 128, -127, 127), b0=q(f['b0'], 16384, -2**31, 2**31 - 1),
                W1=q(f['W1'], 64, -127, 127), b1=q(f['b1'], 8192, -2**31, 2**31 - 1),
                W2=q(f['W2'], 128, -127, 127), b2=q(f['b2'], 16384, -2**31, 2**31 - 1))


if __name__ == '__main__':
    import sys
    pre, st = read_stacks(sys.argv[1])
    print('hashes', {hex(int(d['hash'])) for d in st})
    print('W0 range', min(d['W0'].min() for d in st), max(d['W0'].max() for d in st),
          'b0 range', min(d['b0'].min() for d in st), max(d['b0'].max() for d in st))
    write_stacks('/tmp/rt.nnue', pre, st)
    print('roundtrip identical:', open('/tmp/rt.nnue', 'rb').read() == open(sys.argv[1], 'rb').read())
