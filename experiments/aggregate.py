"""Aggregate per-scenario results into summary tables for the paper/letter.

Walks the results directory, reads every ``<hash>/results.csv`` together with its
``config.yaml``, and produces:
  * experiments/summary.csv  -- one row per scenario with mean/rmse/outlier-rate
  * experiments/outlier_table.csv -- BO vs BO+Ref outlier-rate vs epsilon
    (Reviewer #2, Comment 4)

Usage:
    python experiments/aggregate.py [results_dir]
"""

import os
import sys
import glob
import yaml
import numpy as np
import pandas as pd

from gnss_func.metrics import summary as err_summary

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    results_dir = (sys.argv[1] if len(sys.argv) > 1
                   else os.path.join(HERE, "..", "results"))
    rows = []
    for cfg_path in glob.glob(os.path.join(results_dir, "*", "config.yaml")):
        d = os.path.dirname(cfg_path)
        csv_path = os.path.join(d, "results.csv")
        if not os.path.exists(csv_path):
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        df = pd.read_csv(csv_path)
        s = err_summary(df["error_m"].values)
        rows.append({
            "config_hash": cfg.get("config_hash"),
            "estimator": cfg["estimator"]["name"],
            "delta_tau_frac": cfg["experiment"]["delta_tau_frac"],
            "epsilon": cfg["system"]["epsilon"],
            "cn0_db": cfg["experiment"]["cn0_db"],
            "n_mc": s["n"],
            "mean_m": s["mean_m"],
            "median_m": s["median_m"],
            "rmse_m": s["rmse_m"],
            "outlier_rate": s["outlier_rate"],
            "mean_estimator_time_s": float(df["estimator_time_s"].mean()),
        })
    if not rows:
        print("no results found under", results_dir)
        return
    summary_df = pd.DataFrame(rows).sort_values(
        ["estimator", "delta_tau_frac", "epsilon"])
    out = os.path.join(HERE, "summary.csv")
    summary_df.to_csv(out, index=False)
    print("wrote", out, f"({len(summary_df)} scenarios)")

    # Outlier table: BO vs BO+Ref vs epsilon (Reviewer #2 Comment 4)
    piv = summary_df[summary_df.estimator.isin(["BO", "BO+Ref"])]
    if not piv.empty:
        tab = piv.pivot_table(index="epsilon", columns="estimator",
                              values="outlier_rate", aggfunc=np.mean)
        tab.to_csv(os.path.join(HERE, "outlier_table.csv"))
        print("wrote outlier_table.csv")
        print(tab)


if __name__ == "__main__":
    main()
