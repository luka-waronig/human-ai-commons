"""Fit offset-aware damped oscillations separately to each realization.

python -m plotting.peak_envelope_oscillations --input <sweep.csv>
Positive damping_rate means decay; negative means growth. A finite-window fit
is descriptive and does not establish a limit cycle or asymptotic stability.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from water_toc.reproducibility import write_manifest
from water_toc.timeseries import estimate_oscillation


def analyze_runs(frame, transient_fraction=0.1):
    if not 0 <= transient_fraction < 1:
        raise ValueError('transient_fraction must lie in [0, 1)')
    frame = frame.rename(columns={'Step': 'step'})
    required = {'RunId', 'step', 'Coop_Fraction'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    records = []
    for run_id, group in frame.groupby('RunId', sort=False):
        group = group.sort_values('step')
        if group.step.duplicated().any():
            raise ValueError(f'Duplicate timesteps in run {run_id}')
        group = group[group.step > 0]
        tail = group.iloc[int(transient_fraction * len(group)):]
        result = estimate_oscillation(tail.step.to_numpy(), tail.Coop_Fraction.to_numpy())
        result['RunId'] = run_id
        for name in ['theta', 'max_water_capacity', 'water_cell_density', 'seed', 'feedback_rule']:
            if name in group:
                result[name] = group[name].iloc[0]
        records.append(result)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('water_toc_sweep_results.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/corrected_analysis'))
    args = parser.parse_args()
    results = analyze_runs(pd.read_csv(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'oscillation_fits_per_run.csv', index=False)
    selected = results[results.status == 'fit']
    keys = [key for key in ['theta', 'max_water_capacity', 'water_cell_density', 'feedback_rule'] if key in results]
    if keys and not selected.empty:
        summary = selected.groupby(keys, dropna=False).agg(
            n_fits=('RunId', 'count'), median_period=('period_est', 'median'),
            median_damping=('damping_rate', 'median'),
        ).reset_index()
        totals = results.groupby(keys, dropna=False).size().reset_index(name='n_runs')
        summary = summary.merge(totals, on=keys)
        summary['fit_fraction'] = summary.n_fits / summary.n_runs
        summary.to_csv(args.output_dir / 'oscillation_fit_summary.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, column, label in zip(axes, ['period_est', 'damping_rate'],
                                 ['Period (steps)', 'Decay rate (negative = growth)']):
        ax.hist(selected[column].dropna(), bins=16)
        ax.set(xlabel=label, ylabel='Accepted per-run fits')
    fig.tight_layout()
    fig.savefig(args.output_dir / 'oscillation_fits.png', dpi=200)
    plt.close(fig)
    write_manifest(args.output_dir / 'oscillation_manifest.json',
                   {'analysis': 'offset-aware-per-run-oscillation-v2', 'transient_fraction': 0.1}, input_path=args.input)


if __name__ == '__main__':
    main()
