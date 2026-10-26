"""Compile trainer metric JSONs (+ optional Elo table from ratings.py output) into EXPERIMENTS.csv rows."""
import csv, glob, json, os, re, sys

def ratings(path):
    out = {}
    if path and os.path.exists(path):
        for line in open(path):
            m = re.match(r"(\S+)\s+(-?[\d.]+)\s+\[\s*(-?[\d.]+),\s*(-?[\d.]+)\]", line)
            if m: out[m.group(1)] = (float(m.group(2)), float(m.group(3)), float(m.group(4)))
    return out

def main():
    exp, netdir, rat, hyp, anchor = sys.argv[1:6]
    rt = ratings(rat)
    rows = []
    for f in sorted(glob.glob(netdir + "/*.stud.json")):
        j = json.load(open(f))
        name = os.path.basename(f)[:-10]
        v = j.get("val", {})
        e = rt.get(name)
        rows.append({"id": f"{exp}:{name}", "date": "2026-10-26", "hypothesis": hyp, "code_rev": "see git log",
            "config": f"H={j['H']} label={j['label']} src={j['src']} lr={j['lr']} batch={j['batch']} frac={j.get('frac',1)}",
            "data": ";".join(os.path.basename(x) for x in j["train_files"]), "train_positions": j["positions"],
            "wall_s": j["wall_s"], "cpu_s": j["cpu_s"], "peak_rss_mb": j["peak_rss_mb"],
            "val_game_mse": v.get("vgame", {}).get("mse_sig", ""), "val_tree_mse": v.get("vtree", {}).get("mse_sig", ""),
            "match_opponent": f"joint fit anchor={anchor}" if e else "", "match_cond": "nodes=20000 UHO_4060_v4 paired" if e else "",
            "games": "", "elo": e[0] if e else "", "elo_err95": f"[{e[1]},{e[2]}]" if e else "",
            "notes": f"n_unique={j['n_train_unique']}" + (f" vstree={v['vstree']['mse_sig']}" if 'vstree' in v else "")})
    w = csv.DictWriter(open("/home/user/scratchpad/EXPERIMENTS.csv", "a"), fieldnames=list(rows[0].keys()) if rows else [])
    for r in rows: w.writerow(r)
    print(len(rows), "rows")

main()
