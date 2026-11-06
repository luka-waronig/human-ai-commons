
import itertools
import pandas as pd
from datetime import datetime
import hashlib
import argparse
from pathlib import Path

from water_toc.model import WaterToC
from water_toc.feedback import FEEDBACK_RULES
from water_toc.reproducibility import model_fingerprint, run_identifier, write_manifest

"""
Run a full parameter sweep of the WaterToC model, collecting per-step time series.
For each (theta, Kmax, rho_w) across N_RUNS seeds:
- simulate N_STEPS steps,
- extract required metrics from the model DataCollector,
- attach sweep metadata, and
- write all runs to a single CSV (OUTCSV).
"""

THETAS    = [1.4, 2.0, 5.0, 10]
KMAXES    = [10, 20, 30, 40]
DENSITIES = [0.2, 0.4, 0.5, 0.7]

N_RUNS    = 100
N_STEPS   = 100
OUTCSV    = "water_toc_sweep_results.csv"

BASE_SEED = 42
HEIGHT, WIDTH = 20, 20
INITIAL_HUMANS = 50
INITIAL_AI     = 50

META_COLS = ["RunId","iteration","theta","max_water_capacity","water_cell_density","seed","feedback_rule","model_code_sha256","Step"]
METRIC_COLS = [
    "Coop_Fraction",
    "AI_Coop_Fraction",
    "Human_Coop_Fraction",
    "Environment_State",
    "Total_Water",
    "Avg_Water_Per_Cell",
    "Requested_Water", "Delivered_Water", "Unmet_Water",
]

def deterministic_seed(*args) -> int:
    """
    Create a deterministic integer seed from input arguments.
    Args:
        *args: Values (hashable) that define the seed (e.g., theta, Kmax, rho_w, run index).
    Returns:
        Integer seed derived from BASE_SEED and a SHA1 hash of args.
    """
    s = "_".join(map(str, args))
    return BASE_SEED + int(hashlib.sha1(s.encode()).hexdigest(), 16) % 10_000_000

def run_single_timeseries(theta, kmax, rho_w, run_idx, *, n_steps=None, feedback_rule="repository"):
    """
    Execute one simulation run and return a per-step time-series DataFrame.
    Args:
        theta (float): Feedback strength parameter.
        kmax (int): Maximum water capacity.
        rho_w (float): Water cell density.
        run_idx (int): Index of the run (0-based).
    Returns:
        pandas.DataFrame: Columns META_COLS + METRIC_COLS for all steps of this run.
    Raises:
        ValueError: If required metric columns are missing from the DataCollector.
    """
    seed = deterministic_seed(theta, kmax, rho_w, run_idx)
    n_steps = N_STEPS if n_steps is None else n_steps
    model = WaterToC(
        height=HEIGHT, width=WIDTH,
        initial_humans=INITIAL_HUMANS, initial_ai=INITIAL_AI,
        max_water_capacity=kmax, water_cell_density=rho_w,
        theta=theta, deviation_rate=0.1,
        save_snapshots=False, snapshot_interval=99999,
        seed=seed, feedback_rule=feedback_rule, record_spatial=False,
    )
    for _ in range(n_steps):
        model.step()
    df = model.datacollector.get_model_vars_dataframe().copy()
    if "Step" not in df.columns:
        try:
            df = df.reset_index(names="Step")
        except TypeError:
            df = df.reset_index().rename(columns={"index": "Step"})
    else:
        df = df.reset_index(drop=True)
    fingerprint = model_fingerprint()
    df["RunId"] = run_identifier(dict(theta=theta, capacity=kmax, density=rho_w,
        seed=seed, steps=n_steps, feedback_rule=feedback_rule,
        width=WIDTH, height=HEIGHT, humans=INITIAL_HUMANS, ai=INITIAL_AI,
        model_code_sha256=fingerprint))
    df["iteration"]          = run_idx + 1
    df["theta"]              = theta
    df["max_water_capacity"] = kmax
    df["water_cell_density"] = rho_w
    df["seed"]               = seed
    df["feedback_rule"]      = feedback_rule
    df["model_code_sha256"]  = fingerprint
    missing = [c for c in METRIC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Expected metrics missing from DataCollector: {missing}")
    df = df[META_COLS + METRIC_COLS]
    return df

def main():
    """
    Enumerate all parameter combinations, run N_RUNS replicates each,
    concatenate per-step time series, and save to OUTCSV.
    Outputs:
        - OUTCSV (str): CSV with META_COLS + METRIC_COLS for all runs and steps.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=int, default=N_RUNS)
    parser.add_argument('--steps', type=int, default=N_STEPS)
    parser.add_argument('--theta', type=float, nargs='+', default=THETAS)
    parser.add_argument('--capacity', type=float, nargs='+', default=KMAXES)
    parser.add_argument('--density', type=float, nargs='+')
    parser.add_argument('--design', choices=['repository','manuscript-table2'], default='repository')
    parser.add_argument('--feedback-rule', choices=FEEDBACK_RULES, default='repository')
    parser.add_argument('--output', type=Path, default=Path(OUTCSV))
    args = parser.parse_args()
    if args.runs < 1 or args.steps < 1:
        parser.error('runs and steps must be positive')
    densities = args.density or (DENSITIES if args.design == 'repository' else [.2,.3,.4,.5])
    combos = list(itertools.product(args.theta, args.capacity, densities))
    total_jobs = len(combos) * args.runs
    print(f"Starting sweep: {len(combos)} combos × {args.runs} runs = {total_jobs} sims")
    parts = []
    job = 0
    for theta, kmax, rho_w in combos:
        for r in range(args.runs):
            job += 1
            if job % 10 == 0 or job == 1:
                ts = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] {job}/{total_jobs} | θ={theta}, Kmax={kmax}, ρw={rho_w}, run={r+1}")
            parts.append(run_single_timeseries(theta, kmax, rho_w, r,
                n_steps=args.steps, feedback_rule=args.feedback_rule))
    out = pd.concat(parts, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    write_manifest(args.output.with_suffix('.manifest.json'),
        {**vars(args), 'output':str(args.output), 'density':densities,
         'base_seed':BASE_SEED, 'width':WIDTH, 'height':HEIGHT,
         'humans':INITIAL_HUMANS, 'ai':INITIAL_AI,
         'seed_scheme':'historical SHA1 of stringified theta/capacity/density/replicate'},
        input_path=args.output)
    print(f"Saved {len(out)} rows to {args.output}")

if __name__ == "__main__":
    main()
