"""Aggregate a sim_fast SLURM campaign into the figure CSVs used by the paper.

Reads every ``results_fast/<hash>/{config.yaml,results.csv}`` produced by
``runner_fast.py``, builds a tidy per-scenario summary, and then slices it into
the wide CSVs that the pgfplots files in ``paper/plots/`` expect (same column
names as ``paper/plots/data_revision/*.csv``).

Each campaign realization already contains all four methods (LSKRF, LSKRF+Ref,
BO, BO+Ref), so aggregation is just: group by scenario -> mean/median/outlier.

Usage:
    python aggregate_fast.py [results_dir] [--to-paper]
        results_dir : default ./results_fast
        --to-paper  : also copy the figure CSVs into the paper repo
                      (paper/plots/data_revision/) if that path exists.
"""

import os
import sys
import glob
import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
THR = 5.0  # outlier threshold (m)
B, T = 1.023e6, 1e-3
SNR_OFFSET = -10 * np.log10(2 * B) + 10 * np.log10(B * T)  # snr = cn0 + offset

METHODS = {"lskrf": "err_lskrf", "lskrf_ref": "err_lskrf_ref",
           "bo": "err_bo", "bo_ref": "err_bo_ref"}
# column-name prefix used by the pgfplots (.tikz.tex) files
PFX = {"lskrf": "lskrf", "lskrf_ref": "lskrfref", "bo": "bo", "bo_ref": "boref"}


def summarize(results_dir):
    rows = []
    # new layout: results_fast/<analysis>/<hash>.yaml + <hash>.csv
    for cfgp in glob.glob(os.path.join(results_dir, "*", "*.yaml")):
        csvp = cfgp[:-5] + ".csv"
        if not os.path.exists(csvp):
            continue
        with open(cfgp) as f:
            cfg = yaml.safe_load(f)
        df = pd.read_csv(csvp)
        sc = cfg.get("scenario", {})
        row = {"analysis": cfg.get("analysis"),
               "config_hash": cfg.get("config_hash"), "n_mc": len(df),
               "cn0_db": sc.get("cn0_db"), "angle_diff_deg": sc.get("angle_diff_deg"),
               "delay_diff": sc.get("delay_diff"), "epsilon": sc.get("epsilon"),
               "xi": sc.get("xi"),
               "bo_engine": cfg.get("estimator", {}).get("bo_engine")}
        row["snr_post_db"] = round(row["cn0_db"] + SNR_OFFSET, 2)
        if "i_max" in df.columns:
            row["i_max"] = int(df["i_max"].iloc[0])
        for m, col in METHODS.items():
            e = df[col].values
            row[f"{m}_mean"] = float(np.mean(e))
            row[f"{m}_median"] = float(np.median(e))
            row[f"{m}_out"] = float(np.mean(e > THR))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["epsilon", "angle_diff_deg", "delay_diff", "cn0_db"]).reset_index(drop=True)


def _slice(s, fixed):
    sub = s.copy()
    for k, v in fixed.items():
        if k not in sub.columns:
            return sub.iloc[0:0]
        sub = sub[np.isclose(sub[k].astype(float), float(v))]
    return sub


def figure_csv(summary, x, fixed, out, stat="mean", also_out=True):
    """Wide CSV: x-axis + per-method <pfx>_<stat> (+ <pfx>_out). Column names
    match the pgfplots files (lskrf_mean, lskrfref_mean, bo_mean, boref_mean,...)."""
    sub = _slice(summary, fixed).sort_values(x)
    if sub.empty:
        print(f"  [skip] {os.path.basename(out)}: no rows for {fixed}")
        return False
    cols = {x: sub[x].values}
    for m in ["lskrf", "lskrf_ref", "bo", "bo_ref"]:
        cols[f"{PFX[m]}_{stat}"] = sub[f"{m}_{stat}"].values
        if also_out:
            cols[f"{PFX[m]}_out"] = sub[f"{m}_out"].values
    pd.DataFrame(cols).to_csv(out, index=False)
    print(f"  wrote {os.path.basename(out)} ({len(sub)} pts, fixed={fixed})")
    return True


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    results_dir = args[0] if args else os.path.join(HERE, "results_fast")
    to_paper = "--to-paper" in sys.argv

    summary = summarize(results_dir)
    if summary.empty:
        print("no results under", results_dir)
        return
    sumpath = os.path.join(HERE, "summary_fast.csv")
    summary.to_csv(sumpath, index=False)
    print(f"wrote {sumpath} ({len(summary)} scenarios)")

    # Reference operating point for the slices (edit to taste / after the campaign).
    cn0_ref = 48.0
    dphi_ref = float(summary["angle_diff_deg"].mode().iloc[0])
    dtau_ref = 0.5 if (np.isclose(summary["delay_diff"], 0.5)).any() else \
        float(summary["delay_diff"].mode().iloc[0])
    eps_ref = 0.005 if (np.isclose(summary["epsilon"], 0.005)).any() else \
        float(summary["epsilon"].mode().iloc[0])
    xi_ref = float(summary["xi"].mode().iloc[0])

    outdir = os.path.join(HERE, "figure_csv")
    os.makedirs(outdir, exist_ok=True)
    print(f"slicing at cn0={cn0_ref}, dphi={dphi_ref}, dtau={dtau_ref}, "
          f"eps={eps_ref}, xi={xi_ref}")

    # epsilon sweep (LSKRF / LSKRF+Ref / BO / BO+Ref)
    figure_csv(summary, "epsilon",
               {"cn0_db": cn0_ref, "angle_diff_deg": dphi_ref,
                "delay_diff": dtau_ref, "xi": xi_ref},
               os.path.join(outdir, "R1C4_lskrfref.csv"))
    # angular-separation sweep (operating point where LSKRF fails)
    figure_csv(summary, "angle_diff_deg",
               {"cn0_db": cn0_ref, "delay_diff": dtau_ref,
                "epsilon": eps_ref, "xi": xi_ref},
               os.path.join(outdir, "R1C4_dphi.csv"))
    # SNR sweep
    figure_csv(summary, "snr_post_db",
               {"angle_diff_deg": dphi_ref, "delay_diff": dtau_ref,
                "epsilon": eps_ref, "xi": xi_ref},
               os.path.join(outdir, "R4C5_snr.csv"))
    # I_max sweep (only if the campaign swept i_max)
    if "i_max" in summary.columns:
        figure_csv(summary, "i_max",
                   {"cn0_db": cn0_ref, "angle_diff_deg": dphi_ref,
                    "delay_diff": dtau_ref, "epsilon": eps_ref, "xi": xi_ref},
                   os.path.join(outdir, "R1C3_imax.csv"))
    # xi sweep (median), only if more than one xi
    if summary["xi"].nunique() > 1:
        figure_csv(summary, "xi",
                   {"cn0_db": cn0_ref, "angle_diff_deg": dphi_ref,
                    "delay_diff": dtau_ref, "epsilon": eps_ref},
                   os.path.join(outdir, "R1C5_xi.csv"), stat="median")

    if to_paper:
        dest = os.path.normpath(os.path.join(
            HERE, "..", "..", "..",
            "Documentos/UnB/Artigos/paper-bayesian-gnss/paper/plots/data_revision"))
        if os.path.isdir(dest):
            import shutil
            for f in glob.glob(os.path.join(outdir, "*.csv")):
                shutil.copy(f, dest)
            print("copied figure CSVs ->", dest)
        else:
            print("paper data_revision not found at", dest)


if __name__ == "__main__":
    main()
