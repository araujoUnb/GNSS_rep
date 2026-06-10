"""Auditable, resumable SLURM campaign runner for the FAST original simulator.

Mirrors the gnss_func/experiments design but drives ``sim_fast.FastDelaySim``
(the efficient re-structuring of the ORIGINAL code, faithful to results.dat).

  * A *scenario* = (cn0, angle_diff_deg, delay_diff, epsilon, xi). Scenarios are
    the cartesian product of the lists in ``grid_fast.yaml``; the deterministic
    ordering maps a SLURM array index to a fixed scenario.
  * Each scenario builds ONE FastDelaySim (heavy operators precomputed once) and
    streams ``n_mc`` realizations (seed = base_seed + mc_index).
  * A content hash of the full scenario config names the output dir; the config
    is dumped to config.yaml (audit) and results stream to results.csv every
    ``checkpoint_every`` iterations. Re-running resumes from the first missing
    mc_index (no rework).

Usage:
    python runner_fast.py <scenario_index>     # one SLURM array task
    python runner_fast.py                       # uses $SLURM_ARRAY_TASK_ID
    python runner_fast.py --list                # print scenario count + map
"""

import os
import sys
import json
import time
import hashlib
import itertools

import numpy as np
import pandas as pd
import yaml

from sim_fast import FastDelaySim

HERE = os.path.dirname(os.path.abspath(__file__))


def load_grid(path=None):
    if path is None:
        path = os.environ.get("GNSS_FAST_GRID",
                              os.path.join(HERE, "grid_fast.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def build_scenarios(grid):
    e = grid["experiment"]
    # optional I_max axis (total BO budget); None -> use the original default (62)
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
            "base_seed": int(e["base_seed"]), "n_mc": int(e["n_mc"]),
            "checkpoint_every": int(e["checkpoint_every"]),
        })
    return out


def _hash(cfg):
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def run_scenario(sc, grid, out_root):
    sysp = grid["system"]
    bo_engine = grid["estimator_params"].get("bo_engine", "bayesopt")
    bo_n_init = 2
    n_iter = (62 if sc.get("i_max") is None else int(sc["i_max"])) - bo_n_init
    config = {"system": sysp, "scenario": {k: sc[k] for k in
              ("cn0_db", "angle_diff_deg", "delay_diff", "epsilon", "xi",
               "i_max", "base_seed")},
              "bo_engine": bo_engine,
              "seed_rule": "seed = base_seed + mc_index"}
    h = _hash(config)
    out_dir = os.path.join(out_root, h)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "results.csv")
    yaml_path = os.path.join(out_dir, "config.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            yaml.safe_dump({**config, "config_hash": h}, f, sort_keys=False)

    done = set()
    if os.path.exists(csv_path):
        try:
            done = set(pd.read_csv(csv_path)["mc_index"].astype(int))
        except Exception:
            done = set()

    sim = FastDelaySim(sc["cn0_db"], m=sysp["n_antennas"],
                       k=sysp["n_epochs"], q=sysp["n_correlators"],
                       grid_n=grid["estimator_params"]["grid_n"],
                       bo_engine=bo_engine,
                       bo_n_init=bo_n_init, bo_n_iter=n_iter)
    print(f"[{h}] device={getattr(sim, 'device', 'cpu')} "
          f"n_mc={sc['n_mc']} (resume {len(done)} done)", flush=True)

    buf = []
    for j in range(sc["n_mc"]):
        if j in done:
            continue
        r = sim.run_realization(sc["base_seed"] + j, sc["angle_diff_deg"],
                                sc["delay_diff"], sc["epsilon"], sc["xi"])
        r.update({"mc_index": j, "config_hash": h,
                  "cn0_db": sc["cn0_db"], "angle_diff_deg": sc["angle_diff_deg"],
                  "delay_diff": sc["delay_diff"], "epsilon": sc["epsilon"],
                  "xi": sc["xi"]})
        buf.append(r)
        if len(buf) >= sc["checkpoint_every"]:
            pd.DataFrame(buf).to_csv(csv_path, mode="a", index=False,
                                     header=not os.path.exists(csv_path))
            buf = []
            print(f"[{h}] checkpoint mc_index={j}", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(csv_path, mode="a", index=False,
                                 header=not os.path.exists(csv_path))
    print(f"[{h}] done -> {csv_path}", flush=True)
    return csv_path


def main():
    grid = load_grid()
    scs = build_scenarios(grid)
    if "--list" in sys.argv:
        print(f"{len(scs)} scenarios (SLURM array 0-{len(scs) - 1})")
        for i, s in enumerate(scs):
            print(i, "cn0=%.0f" % s["cn0_db"], "adeg=%.1f" % s["angle_diff_deg"],
                  "dd=%.2f" % s["delay_diff"], "eps=%.3f" % s["epsilon"],
                  "xi=%.3f" % s["xi"])
        return
    idx = (int(sys.argv[1]) if len(sys.argv) > 1
           else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))
    if not (0 <= idx < len(scs)):
        raise SystemExit(f"index {idx} out of range 0..{len(scs)-1}")
    out_root = os.path.join(HERE, grid.get("out_root", "results_fast"))
    t = time.perf_counter()
    run_scenario(scs[idx], grid, out_root)
    print(f"[scenario {idx}] wall {time.perf_counter()-t:.1f}s", flush=True)


if __name__ == "__main__":
    main()
