"""Auditable, resumable campaign runner for the FAST original simulator.

Design (per the project conventions):
  * A *campaign/analysis* is one YAML grid (e.g. campaigns/eps.yaml) with a
    ``name`` and one varying axis at a reference operating point. It maps to ONE
    SLURM array job (job-name = analysis name) so it is easy to see which
    analysis has finished.
  * A *scenario* is one point of the grid's cartesian product. Its output files
    are named by a HASH of ALL simulation parameters (system + scenario +
    estimator + base_seed): ``<out>/<name>/<hash>.csv`` and the identically-named
    ``<out>/<name>/<hash>.yaml`` that stores those parameters (audit). n_mc and
    checkpoint_every are NOT in the hash, so you can extend n_mc and resume.
  * Results stream to the CSV every ``checkpoint_every`` (=10) realizations;
    seed = base_seed + mc_index; re-running resumes from the first missing index.

Usage:
    python runner_fast.py <i>                 # run scenario i of $GNSS_FAST_GRID
    python runner_fast.py                      # uses $SLURM_ARRAY_TASK_ID
    python runner_fast.py --list               # list scenarios
    python runner_fast.py --count              # print scenario count only
Set the grid with GNSS_FAST_GRID=campaigns/<analysis>.yaml (default grid_fast.yaml).
"""

import os
import sys
import glob
import json
import time
import hashlib
import itertools

import pandas as pd
import yaml

from sim_fast import FastDelaySim

HERE = os.path.dirname(os.path.abspath(__file__))


def load_grid(path=None):
    if path is None:
        path = os.environ.get("GNSS_FAST_GRID",
                              os.path.join(HERE, "grid_fast.yaml"))
    with open(path) as f:
        g = yaml.safe_load(f)
    g.setdefault("name", os.path.splitext(os.path.basename(path))[0])
    return g


def build_scenarios(grid):
    e = grid["experiment"]
    i_max_list = e.get("i_max", [None])
    combos = itertools.product(
        e["cn0_db"], e["angle_diff_deg"], e["delay_diff"],
        e["epsilon"], e["xi"], i_max_list)
    out = []
    for cn0, adeg, dd, eps, xi, imax in combos:
        out.append({
            "cn0_db": float(cn0), "angle_diff_deg": float(adeg),
            "delay_diff": float(dd), "epsilon": float(eps), "xi": float(xi),
            "i_max": (None if imax is None else int(imax)),
        })
    return out


def all_campaigns(exclude=("smoke",)):
    """Flatten every campaigns/*.yaml into one ordered list of (grid, scenario),
    so a SINGLE SLURM array (with %MAXP) can run the whole study under a global
    concurrency cap. 'smoke' is excluded."""
    flat = []
    for gp in sorted(glob.glob(os.path.join(HERE, "campaigns", "*.yaml"))):
        name = os.path.splitext(os.path.basename(gp))[0]
        if name in exclude:
            continue
        grid = load_grid(gp)
        for sc in build_scenarios(grid):
            flat.append((grid, sc))
    return flat


def scenario_params(sc, grid):
    """Every parameter that DEFINES the scenario (-> hash). Excludes n_mc."""
    sysp = grid["system"]
    ep = grid["estimator_params"]
    return {
        "system": {"n_antennas": sysp["n_antennas"],
                   "n_epochs": sysp["n_epochs"],
                   "n_correlators": sysp["n_correlators"]},
        "scenario": sc,
        "estimator": {"grid_n": ep["grid_n"],
                      "objective": ep.get("objective", "exact"),
                      "bo_engine": ep.get("bo_engine", "bayesopt")},
        "base_seed": int(grid["experiment"]["base_seed"]),
    }


def _hash(params):
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def run_scenario(sc, grid, out_root):
    sysp, ep = grid["system"], grid["estimator_params"]
    exp = grid["experiment"]
    params = scenario_params(sc, grid)
    h = _hash(params)
    n_mc = int(exp["n_mc"])
    checkpoint = int(exp.get("checkpoint_every", 10))

    out_dir = os.path.join(out_root, grid["name"])
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{h}.csv")
    yaml_path = os.path.join(out_dir, f"{h}.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            yaml.safe_dump({**params, "analysis": grid["name"],
                            "n_mc": n_mc, "checkpoint_every": checkpoint,
                            "config_hash": h,
                            "seed_rule": "seed = base_seed + mc_index"},
                           f, sort_keys=False)

    done = set()
    if os.path.exists(csv_path):
        try:
            done = set(pd.read_csv(csv_path)["mc_index"].astype(int))
        except Exception:
            done = set()

    bo_n_init = 2
    n_iter = (62 if sc["i_max"] is None else sc["i_max"]) - bo_n_init
    sim = FastDelaySim(sc["cn0_db"], m=sysp["n_antennas"], k=sysp["n_epochs"],
                       q=sysp["n_correlators"], grid_n=ep["grid_n"],
                       bo_engine=ep.get("bo_engine", "bayesopt"),
                       objective=ep.get("objective", "exact"),
                       bo_n_init=bo_n_init, bo_n_iter=n_iter)
    print(f"[{grid['name']}/{h}] device={sim.device} n_mc={n_mc} "
          f"(resume {len(done)} done)", flush=True)

    base = params["base_seed"]
    buf = []
    for j in range(n_mc):
        if j in done:
            continue
        r = sim.run_realization(base + j, sc["angle_diff_deg"],
                                sc["delay_diff"], sc["epsilon"], sc["xi"])
        r.update({"mc_index": j, "config_hash": h, "analysis": grid["name"],
                  "cn0_db": sc["cn0_db"], "angle_diff_deg": sc["angle_diff_deg"],
                  "delay_diff": sc["delay_diff"], "epsilon": sc["epsilon"],
                  "xi": sc["xi"]})
        buf.append(r)
        if len(buf) >= checkpoint:
            pd.DataFrame(buf).to_csv(csv_path, mode="a", index=False,
                                     header=not os.path.exists(csv_path))
            buf = []
            print(f"[{grid['name']}/{h}] checkpoint mc_index={j}", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(csv_path, mode="a", index=False,
                                 header=not os.path.exists(csv_path))
    print(f"[{grid['name']}/{h}] done -> {csv_path}", flush=True)
    return csv_path


def main():
    # Combined mode: GNSS_FAST_ALL=1 runs ALL campaigns as one flat index space
    # (one SLURM array with %MAXP for a global concurrency cap).
    if os.environ.get("GNSS_FAST_ALL") == "1":
        flat = all_campaigns()
        if "--count" in sys.argv:
            print(len(flat))
            return
        if "--list" in sys.argv:
            print(f"[ALL] {len(flat)} scenarios (array 0-{len(flat)-1})")
            for i, (g, s) in enumerate(flat):
                print(i, g["name"], {k: s[k] for k in
                      ("cn0_db", "angle_diff_deg", "delay_diff",
                       "epsilon", "xi", "i_max")})
            return
        idx = (int(sys.argv[1]) if len(sys.argv) > 1
               and not sys.argv[1].startswith("-")
               else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
        if not (0 <= idx < len(flat)):
            raise SystemExit(f"index {idx} out of range 0..{len(flat)-1}")
        grid, sc = flat[idx]
        out_root = os.path.join(HERE, grid.get("out_root", "results_fast"))
        t = time.perf_counter()
        run_scenario(sc, grid, out_root)
        print(f"[ALL {grid['name']} scenario {idx}] "
              f"wall {time.perf_counter()-t:.1f}s", flush=True)
        return

    grid = load_grid()
    scs = build_scenarios(grid)
    if "--count" in sys.argv:
        print(len(scs))
        return
    if "--list" in sys.argv:
        print(f"[{grid['name']}] {len(scs)} scenarios (array 0-{len(scs)-1})")
        for i, s in enumerate(scs):
            print(i, {k: s[k] for k in
                      ("cn0_db", "angle_diff_deg", "delay_diff", "epsilon",
                       "xi", "i_max")})
        return
    idx = (int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
           else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if not (0 <= idx < len(scs)):
        raise SystemExit(f"index {idx} out of range 0..{len(scs)-1}")
    out_root = os.path.join(HERE, grid.get("out_root", "results_fast"))
    t = time.perf_counter()
    run_scenario(scs[idx], grid, out_root)
    print(f"[{grid['name']} scenario {idx}] wall {time.perf_counter()-t:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
