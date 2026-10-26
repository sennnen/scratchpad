"""Joint Elo fit (Bradley-Terry on game-pair scores) over all match JSONs in given dirs.
Uncertainty: bootstrap resampling of game pairs (pentanomial) within each match.
Usage: ratings.py --anchor NAME dir1 [dir2 ...]"""
import argparse, glob, json, re
import numpy as np


def players(cmd):
    return [c.split("=", 1)[1] for c in cmd if c.startswith("name=")]


def fit(matches, names, anchor, iters=3000):
    idx = {n: i for i, n in enumerate(names)}
    r = np.zeros(len(names))
    for _ in range(iters):
        g = np.zeros(len(names)); h = np.zeros(len(names))
        for a, b, score, n in matches:
            p = 1 / (1 + 10 ** ((r[idx[b]] - r[idx[a]]) / 400))
            d = (score - n * p)
            g[idx[a]] += d; g[idx[b]] -= d
            w = n * p * (1 - p) * np.log(10) / 400
            h[idx[a]] += w; h[idx[b]] += w
        step = 400 / np.log(10) * g / np.maximum(h * 400 / np.log(10), 1e-9)
        r += np.clip(step, -50, 50)
        r -= r[idx[anchor]]
        if np.abs(step).max() < 1e-4: break
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--anchor", required=True)
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--nodes", type=int, default=None)
    a = ap.parse_args()
    ms = []
    for d in a.dirs:
        for f in sorted(glob.glob(d + "/*.json")):
            j = json.load(open(f))
            if "ptnml" not in j: continue
            if a.nodes and f"nodes={a.nodes}" not in j["cmd"]: continue
            pa, pb = players(j["cmd"])
            ms.append((pa, pb, np.array(j["ptnml"])))
    names = sorted({x for m in ms for x in m[:2]})
    def to_scores(ptn_list):
        return [(pa, pb, float((p * np.arange(5)).sum() / 2), float(p.sum() * 2)) for (pa, pb, _), p in zip(ms, ptn_list)]
    base = fit(to_scores([m[2] for m in ms]), names, a.anchor)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        pl = []
        for _, _, p in ms:
            n = p.sum()
            pl.append(rng.multinomial(n, p / n))
        boots.append(fit(to_scores(pl), names, a.anchor, iters=500))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)
    order = np.argsort(-base)
    print(f"{'player':32s} {'elo':>8s} {'95% CI':>18s}")
    for i in order:
        print(f"{names[i]:32s} {base[i]:8.1f}   [{lo[i]:7.1f}, {hi[i]:7.1f}]")
    print("matches:")
    for pa, pb, p in ms:
        print(f"  {pa} vs {pb}: ptnml {p.tolist()} games {2*p.sum()}")


if __name__ == "__main__":
    main()
