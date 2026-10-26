"""Train a student network on research-fork records. Writes .stud net + JSON metrics."""
import argparse, json, os, resource, subprocess, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(__file__))
from data import load, features
from model import Student, OUT_SCALE


def targets(recs, label, lam):
    t = torch.from_numpy(recs["teacher"].astype(np.float32))
    s = torch.from_numpy(recs["score"].astype(np.float32))
    r = torch.from_numpy(recs["result"].astype(np.float32)) / 2.0
    if label == "teacher":
        return torch.sigmoid(t / OUT_SCALE)
    if label == "score":
        return torch.sigmoid(s / OUT_SCALE)
    if label == "blend":  # conventional: lambda * search score + (1-lambda) * game result
        return lam * torch.sigmoid(s / OUT_SCALE) + (1 - lam) * r
    raise ValueError(label)


@torch.no_grad()
def evaluate(model, recs, bs=65536):
    """MSE vs teacher in sigmoid space, and mean abs error in internal units (clipped)."""
    mse, mae, n = 0.0, 0.0, 0
    for i in range(0, len(recs), bs):
        r = recs[i:i + bs]
        s, nn_, b = features(r)
        o = model(s, nn_, b)
        t = torch.from_numpy(r["teacher"].astype(np.float32))
        mse += ((torch.sigmoid(o) - torch.sigmoid(t / OUT_SCALE)) ** 2).sum().item()
        mae += (o * OUT_SCALE - t).clamp(-2000, 2000).abs().sum().item()
        n += len(r)
    return {"mse_sig": mse / max(n, 1), "mae_iu": mae / max(n, 1), "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--src", type=int, default=None, help="0 game positions, 1 tree positions")
    ap.add_argument("--label", default="teacher", choices=["teacher", "score", "blend"])
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--n", type=int, default=None, help="max training positions")
    ap.add_argument("--positions", type=float, required=True, help="total positions to train on (steps*batch)")
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--H", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val", nargs="*", default=[], help="name=path[:src] validation sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    t0 = time.time()
    c0 = resource.getrusage(resource.RUSAGE_SELF)
    recs = load(a.train, src=a.src)
    rng = np.random.default_rng(a.seed)
    if a.n is not None and a.n < len(recs):
        recs = recs[rng.choice(len(recs), a.n, replace=False)]
    N = len(recs)
    vals = {}
    for v in a.val:
        name, spec = v.split("=")
        path, _, src = spec.partition(":")
        vals[name] = load([path], src=int(src) if src else None, max_n=200000)
    model = Student(H=a.H)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    steps = int(a.positions // a.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.05,
                                                anneal_strategy="cos", div_factor=10, final_div_factor=100)
    perm = rng.permutation(N)
    ptr = 0
    hist = []
    for step in range(steps):
        if ptr + a.batch > N:
            perm = rng.permutation(N)
            ptr = 0
        r = recs[perm[ptr:ptr + a.batch]]
        ptr += a.batch
        s, nn_, b = features(r)
        y = targets(r, a.label, a.lam)
        o = model(s, nn_, b)
        loss = ((torch.sigmoid(o) - y) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % max(1, steps // 10) == 0 or step == steps - 1:
            hist.append((step, loss.item()))
            print(f"step {step}/{steps} loss {loss.item():.6f} {time.time()-t0:.0f}s", flush=True)
    c1 = resource.getrusage(resource.RUSAGE_SELF)
    model.export(a.out)
    res = {
        "args": vars(a), "n_train_unique": int(N), "steps": steps,
        "wall_s": time.time() - t0, "cpu_s": (c1.ru_utime + c1.ru_stime) - (c0.ru_utime + c0.ru_stime),
        "peak_rss_mb": c1.ru_maxrss / 1024, "loss_hist": hist,
        "val": {k: evaluate(model, v) for k, v in vals.items()},
        "git": subprocess.run(["git", "-C", os.path.dirname(__file__), "rev-parse", "HEAD"],
                              capture_output=True, text=True).stdout.strip(),
    }
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(json.dumps(res["val"]), f"wall {res['wall_s']:.0f}s cpu {res['cpu_s']:.0f}s")


if __name__ == "__main__":
    main()
