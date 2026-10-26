"""Fine-tune the 32 dense layer stacks of a production SFNNv18 net on ftdump records.

The feature transformer (110 MB) is frozen: its u8 outputs are precomputed by `ftdump` in the research
fork. The layer stacks (1.13M parameters) are trained with fake quantization that mirrors Stockfish's
integer inference (floor on activations, rounded int8 weights), initialised from the source net, with an
optional L2 anchor to the initial weights. Targets are search scores (Rec.score, stm view) mapped
through the inverse of Stockfish's scale_evaluation (optimism = 0), so that after Stockfish rescales the
raw net output at inference time it matches the search score.

Usage: finetune.py --net master.nnue --train SET.ft[:weight] ... --val name=SET.ft ... --out new.nnue
"""
import argparse, json, os, resource, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from nnue_stacks import read_stacks, write_stacks, to_float, to_quant
from ftdata import load

SIG = 340.0
PV, NV, BV, RV, QV = 208, 781, 825, 1276, 2538


def board_stats(recs):
    """simple_eval (stm view), material term and rule50 for scale_evaluation."""
    b = recs['board']
    codes = np.stack([b & 15, b >> 4], 2).reshape(len(recs), 64).astype(np.int64)
    stm = recs['stm'].astype(np.int64)[:, None]
    val = np.array([0, PV, NV, BV, RV, QV, 0, 0, 0, PV, NV, BV, RV, QV, 0, 0])
    v = val[codes]
    col = codes >> 3
    own = np.where(col == stm, 1, -1) * (codes != 0)
    se = (v * own).sum(1)
    npawn = ((codes & 7) == 1).sum(1)
    nonpawn = (v * ((codes & 7) != 1)).sum(1)
    material = 521 * npawn + nonpawn
    return se, material, recs['rule50'].astype(np.int64)


def wdl_a(recs):
    """Stockfish win_rate_params 'a' (uci.cpp): Score::InternalUnits stores to_cp(v) = round(100 v / a)."""
    b = recs['board']
    codes = np.stack([b & 15, b >> 4], 2).reshape(len(recs), 64).astype(np.int64)
    w = np.array([0, 1, 3, 3, 5, 9, 0, 0, 0, 1, 3, 3, 5, 9, 0, 0])
    m = np.clip(w[codes].sum(1), 17, 78) / 58.0
    a0, a1, a2, a3 = -142.72052667, 372.35176398, -340.71073572, 415.23490212
    return ((a0 * m + a1) * m + a2) * m + a3


def cp_to_internal(cp, recs):
    """Recorded search scores (Rec.score from `label`/`gendata`) are centipawns; convert to Value units."""
    return cp.astype(np.float64) * wdl_a(recs) / 100.0


def scale_eval(nnue, se, material, r50):
    """Vectorised Stockfish scale_evaluation with optimism 0 (integer semantics approximated in float)."""
    se_norm = np.trunc(se * 1024 / (np.abs(se) + 1024))
    nn_norm = np.trunc(nnue * 1024 / (np.abs(nnue) + 1024))
    correl = np.trunc(se_norm * nn_norm / 512)
    v = nnue + np.trunc(nnue * correl / 65536)
    v = v + np.trunc(material * v / 90649)
    v = v - np.trunc(v * r50 / 189)
    return v


def inverse_scale(score, se, material, r50, lo=-12000.0, hi=12000.0, iters=40):
    s = np.clip(score.astype(np.float64), -6000, 6000)
    a = np.full_like(s, lo); b = np.full_like(s, hi)
    for _ in range(iters):
        m = (a + b) / 2
        f = scale_eval(m, se, material, r50)
        a = np.where(f < s, m, a); b = np.where(f < s, b, m)
    return (a + b) / 2


def ste_floor(x, s):
    return x + ((x * s + 1e-5).floor() / s - x).detach()


def ste_round(w, s):
    return w + ((w * s).round() / s - w).detach()


class Stacks(torch.nn.Module):
    def __init__(self, st):
        super().__init__()
        f = [to_float(d) for d in st]
        self.hashes = [d['hash'] for d in st]
        P = lambda k: torch.nn.Parameter(torch.tensor(np.stack([x[k] for x in f]), dtype=torch.float32))
        self.W0, self.b0, self.W1, self.b1, self.W2, self.b2 = P('W0'), P('b0'), P('W1'), P('b1'), P('W2'), P('b2')
        self.init = {k: v.detach().clone() for k, v in self.named_parameters()}

    @staticmethod
    def act(h):
        sq = torch.clamp(ste_floor(h * h, 128), max=127 / 128)
        cl = torch.clamp(ste_floor(h, 128), 0, 127 / 128)
        return torch.cat([sq, cl], 1)

    def forward(self, ft, bucket):
        """Batch is processed per bucket on contiguous slices (sorted by bucket)."""
        order = torch.argsort(bucket, stable=True)
        inv = torch.empty_like(order); inv[order] = torch.arange(len(order))
        ft, bucket = ft[order], bucket[order]
        counts = torch.bincount(bucket, minlength=len(self.hashes)).tolist()
        W0 = ste_round(self.W0.clamp(-127 / 128, 127 / 128), 128)
        W1 = ste_round(self.W1.clamp(-127 / 64, 127 / 64), 64)
        W2 = ste_round(self.W2.clamp(-127 / 128, 127 / 128), 128)
        outs, start = [], 0
        for b, c in enumerate(counts):
            if c == 0: continue
            x = ft[start:start + c]; start += c
            h0 = torch.addmm(self.b0[b], x, W0[b].T)
            a0 = self.act(h0)
            h1 = torch.addmm(self.b1[b], a0, W1[b].T)
            a1 = self.act(h1)
            o = torch.addmm(self.b2[b], torch.cat([a0, a1], 1), W2[b].T)
            outs.append(o[:, 0] + h0[:, 30] - h0[:, 31])
        out = torch.cat(outs)[inv]
        return out * 600.0  # raw eval, internal units

    def anchor(self):
        return sum(((p - self.init[k]) ** 2).sum() for k, p in self.named_parameters())

    def export(self, prefix, path):
        st = []
        for b in range(len(self.hashes)):
            f = {k: getattr(self, k)[b].detach().double().numpy() for k in ('W0', 'b0', 'W1', 'b1', 'W2', 'b2')}
            st.append(to_quant(f, self.hashes[b]))
        write_stacks(path, prefix, st)


def prepare(d, quiet_only=False, nkeep=0, target='search'):
    """Returns (ft u8 tensor, bucket, raw-space target, mask of kept rows)."""
    recs = d['rec']
    keep = np.ones(len(d), bool)
    if quiet_only:
        keep &= (recs['src'] & 0x80) == 0
    keep &= np.abs(recs['score']) < 20000  # drop mate/TB scores (stored as +-31000/32000)
    idx = np.nonzero(keep)[0]
    if nkeep:
        idx = idx[:nkeep]
    r = recs[idx]
    if target == 'teacher':  # distillation: the source net's own raw eval (ftdump sfEval)
        t = d['sfeval'][idx].astype(np.float64)
    else:
        se, mat, r50 = board_stats(r)
        t = inverse_scale(cp_to_internal(r['score'], r), se, mat, r50)
    return (torch.from_numpy(np.ascontiguousarray(d['ft'][idx])), torch.from_numpy(d['bucket'][idx].astype(np.int64)),
            torch.from_numpy(t.astype(np.float32)), torch.from_numpy(d['sfeval'][idx].astype(np.float32)))


def loss_fn(pred, t):
    return ((torch.sigmoid(pred / SIG) - torch.sigmoid(t / SIG)) ** 2).mean()


@torch.no_grad()
def evaluate(model, data, bs=32768):
    ft, bk, t, sf = data
    l, l0, n = 0.0, 0.0, 0
    for i in range(0, len(ft), bs):
        p = model(ft[i:i + bs].float() / 128, bk[i:i + bs])
        l += loss_fn(p, t[i:i + bs]).item() * len(p)
        l0 += loss_fn(sf[i:i + bs], t[i:i + bs]).item() * len(p)
        n += len(p)
    return {'mse': l / n, 'mse_orig': l0 / n, 'n': n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--net', required=True)
    ap.add_argument('--train', nargs='+', required=True, help='path[:fraction_of_batch]')
    ap.add_argument('--val', nargs='*', default=[])
    ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=8192)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--anchor', type=float, default=0.0)
    ap.add_argument('--quiet', action='store_true', help='drop positions whose search best move is a capture')
    ap.add_argument('--n', type=int, default=0, help='use only the first n positions of each train set')
    ap.add_argument('--nkeep', type=int, default=0, help='cap each train set at this many rows after filtering')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--target', default='search', choices=['search', 'teacher'])
    ap.add_argument('--reinit', action='store_true', help='random re-initialisation of all layer stacks')
    ap.add_argument('--threads', type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    prefix, st = read_stacks(a.net)
    model = Stacks(st)
    if a.reinit:
        with torch.no_grad():
            g = torch.Generator().manual_seed(a.seed)
            for W, b, fan in ((model.W0, model.b0, 1024), (model.W1, model.b1, 64), (model.W2, model.b2, 128)):
                lim = 1.0 / fan ** 0.5
                W.uniform_(-lim, lim, generator=g); b.uniform_(-lim, lim, generator=g)
            model.init = {k: v.detach().clone() for k, v in model.named_parameters()}
    sets = []
    for spec in a.train:
        path, _, w = spec.partition(':')
        d = load(path)
        if a.n: d = d[:a.n]
        sets.append((prepare(d, a.quiet, a.nkeep, a.target), float(w) if w else 1.0, path))
    tot = sum(w for _, w, _ in sets)
    vals = {}
    for v in a.val:
        name, path = v.split('=')
        vals[name] = prepare(load(path), False, 0, a.target)
    res = {'args': vars(a), 'train_sizes': [len(s[0][0]) for s in sets]}
    res['val_before'] = {k: evaluate(model, v) for k, v in vals.items()}
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps, eta_min=a.lr / 20)
    hist = []
    for step in range(a.steps):
        fts, bks, ts = [], [], []
        for (ft, bk, t, _), w, _ in sets:
            k = int(round(a.batch * w / tot))
            i = torch.from_numpy(rng.integers(0, len(ft), k))
            fts.append(ft[i]); bks.append(bk[i]); ts.append(t[i])
        ft = torch.cat(fts).float() / 128; bk = torch.cat(bks); t = torch.cat(ts)
        loss = loss_fn(model(ft, bk), t)
        tot_loss = loss + (a.anchor * model.anchor() if a.anchor else 0)
        opt.zero_grad(); tot_loss.backward(); opt.step(); sched.step()
        if step % max(1, a.steps // 10) == 0 or step == a.steps - 1:
            hist.append((step, loss.item()))
            print(f'step {step} loss {loss.item():.6f} {time.time()-t0:.0f}s', flush=True)
    model.export(prefix, a.out)
    res['val_after'] = {k: evaluate(model, v) for k, v in vals.items()}
    ru = resource.getrusage(resource.RUSAGE_SELF)
    res.update(wall_s=time.time() - t0, cpu_s=ru.ru_utime + ru.ru_stime, peak_rss_mb=ru.ru_maxrss / 1024, hist=hist,
               anchor_dist=model.anchor().item())
    json.dump(res, open(a.out + '.json', 'w'), indent=1)
    print(json.dumps({k: res[k] for k in ('val_before', 'val_after')}))


if __name__ == '__main__':
    main()
