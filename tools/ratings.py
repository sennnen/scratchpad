"""Joint Elo fit (Bradley-Terry on game-pair scores) over all match JSONs in given dirs.
Uncertainty: bootstrap resampling of game pairs (pentanomial) within each match.
Usage: ratings.py --anchor NAME dir1 [dir2 ...]"""
import argparse, glob, json, re
import numpy as np


def players(cmd):
    return [c.split("=", 1)[1] for c in cmd if c.startswith("name=")]


def fit(matches, names, anchor, iters=200):
    """Bradley-Terry MLE (logistic Elo) by full Newton on the anchored Hessian. The anchor is fixed at 0.
    (Earlier versions used per-player Newton steps including the anchor and re-centred, which can stop
    at a non-optimal fixed point; see RESEARCH_LOG 'BUG FOUND (ratings.py)'.)"""
    idx = {n: i for i, n in enumerate(names)}
    k = len(names); c = np.log(10) / 400
    free = [i for i in range(k) if i != idx[anchor]]
    r = np.zeros(k)
    for _ in range(iters):
        g = np.zeros(k); H = np.zeros((k, k))
        for a, b, score, n in matches:
            i, j = idx[a], idx[b]
            p = 1 / (1 + 10 ** ((r[j] - r[i]) / 400))
            d = c * (score - n * p)
            g[i] += d; g[j] -= d
            w = n * p * (1 - p) * c * c
            H[i, i] += w; H[j, j] += w; H[i, j] -= w; H[j, i] -= w
        Hf = H[np.ix_(free, free)] + 1e-9 * np.eye(len(free))
        step = np.linalg.solve(Hf, g[free])
        step = np.clip(step, -200, 200)
        r[free] += step
        if np.abs(step).max() < 1e-6: break
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--anchor", required=True)
    ap.add_argument("--alias", nargs="*", default=[], help="old=new name merges")
    ap.add_argument("--boot", type=int, default=200); ap.add_argument("--nodes", type=int, default=None)
    a = ap.parse_args()
    ms = []
    for d in a.dirs:
        for f in sorted(glob.glob(d + "/*.json")):
            j = json.load(open(f))
            if "ptnml" not in j: continue
            if a.nodes and f"nodes={a.nodes}" not in j["cmd"]: continue
            pa, pb = players(j["cmd"])
            al = dict(x.split("=") for x in a.alias)
            pa, pb = al.get(pa, pa), al.get(pb, pb)
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
        boots.append(fit(to_scores(pl), names, a.anchor, iters=100))
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
