"""Run a paired-opening match between two engine configurations with fastchess and record results.
Engine spec: name=teacher | name=/path/to/net.stud  (student loaded into the research fork)."""
import argparse, json, os, re, subprocess, time

SF = os.environ.get("SF_BIN", "/home/user/data/sf_research")
FC = os.environ.get("FASTCHESS", "/home/user/work/fastchess/fastchess")


def engine_args(spec, binary, nodes=None, tc=None):
    """spec: name=net[@nodes]  (net 'teacher' = master net; '@nodes' overrides the node budget)"""
    name, _, net = spec.partition("=")
    net, _, n_over = net.partition("@")
    a = ["-engine", f"cmd={binary}", f"name={name}"]
    if net.endswith(".nnue"):
        if not os.path.isfile(net):
            raise SystemExit(f"net not found: {net}")
        a.append(f"option.EvalFile={net}")
    elif net and net != "teacher":
        if not os.path.isfile(net):
            raise SystemExit(f"student net not found: {net} (would silently fall back to the teacher)")
        a.append(f"option.StudentFile={net}")
    if tc:
        a += [f"tc={tc}"]
    else:
        a += ["tc=inf", f"nodes={n_over or nodes}"]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--nodes", type=int, default=20000)
    ap.add_argument("--tc", default=None, help="time control instead of nodes, e.g. 10+0.1")
    ap.add_argument("--rounds", type=int, default=500)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--book", default="/home/user/data/UHO_4060_v4.epd")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--bin_a", default=SF); ap.add_argument("--bin_b", default=SF)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sprt", default=None, help="elo0,elo1")
    a = ap.parse_args()
    cmd = [FC] + engine_args(a.a, a.bin_a, a.nodes, a.tc) + engine_args(a.b, a.bin_b, a.nodes, a.tc) + [
        "-each", "option.Threads=1", "option.Hash=16",
        "-openings", f"file={a.book}", "format=epd", "order=random",
        "-srand", str(a.seed), "-repeat", "-rounds", str(a.rounds), "-concurrency", str(a.concurrency),
        "-resign", "movecount=3", "score=1000", "twosided=true",
        "-draw", "movenumber=40", "movecount=8", "score=8",
        "-pgnout", f"file={a.out}.pgn", "-ratinginterval", "100"]
    if a.sprt:
        e0, e1 = a.sprt.split(",")
        cmd += ["-sprt", f"elo0={e0}", f"elo1={e1}", "alpha=0.05", "beta=0.05", "model=normalized"]
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    txt = p.stdout
    open(a.out + ".log", "w").write(" ".join(cmd) + "\n" + txt + p.stderr)
    blocks = txt.split("Results of")
    last = blocks[-1] if len(blocks) > 1 else txt
    res = {"cmd": cmd, "wall_s": time.time() - t0}
    m = re.search(r"Elo:\s*([-\d.inf]+)\s*\+/-\s*([-\d.naninf]+)", last)
    if m: res["elo"], res["elo_err95"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"nElo:\s*([-\d.inf]+)\s*\+/-\s*([-\d.naninf]+)", last)
    if m: res["nelo"], res["nelo_err95"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)", last)
    if m: res["games"], res["w"], res["l"], res["d"] = map(int, m.groups())
    m = re.search(r"Ptnml\(0-2\):\s*\[([\d,\s]+)\]", last)
    if m: res["ptnml"] = [int(x) for x in m.group(1).split(",")]
    m = re.search(r"LOS:\s*([\d.]+)", last)
    if m: res["los"] = float(m.group(1))
    m = re.search(r"LLR:\s*([-\d.]+)", last)
    if m: res["llr"] = float(m.group(1))
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    print(a.a, "vs", a.b, {k: v for k, v in res.items() if k != "cmd"})


if __name__ == "__main__":
    main()
