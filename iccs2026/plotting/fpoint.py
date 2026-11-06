"""Per-run low-variance diagnostics, not proofs of fixed-point attractors.

Run from iccs2026/: python -m plotting.fpoint --input <sweep.csv>
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from water_toc.reproducibility import write_manifest
from water_toc.timeseries import safe_adf_p, tail_diagnostics


def classify_runs(frame, tail_fraction=0.2):
    if not 0 < tail_fraction <= 1:
        raise ValueError('tail_fraction must lie in (0, 1]')
    frame = frame.rename(columns={'Step': 'step'})
    required = {'RunId', 'step', 'Coop_Fraction', 'Environment_State'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    records = []
    for run_id, group in frame.groupby('RunId', sort=False):
        group = group.sort_values('step')
        if group.step.duplicated().any():
            raise ValueError(f'Duplicate timesteps in run {run_id}')
        post = group[group.step > 0]
        if post.empty:
            raise ValueError(f'Run {run_id} has no post-initialization observations')
        tail = post.iloc[int((1 - tail_fraction) * len(post)):]
        record = {'RunId': run_id, 'tail_points': len(tail)}
        for name in ['theta', 'max_water_capacity', 'water_cell_density', 'seed', 'feedback_rule']:
            if name in group:
                record[name] = group[name].iloc[0]
        for label, column in [('coop', 'Coop_Fraction'), ('env', 'Environment_State')]:
            diagnostics = tail_diagnostics(tail[column].to_numpy())
            record.update({f'{label}_{key}': value for key, value in diagnostics.items()})
            record[f'tail_mean_{label}'] = float(tail[column].mean())
        records.append(record)
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('water_toc_sweep_results.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/corrected_analysis'))
    parser.add_argument('--tail-fraction', type=float, default=0.2)
    args = parser.parse_args()
    results = classify_runs(pd.read_csv(args.input), args.tail_fraction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / 'low_variance_per_run.csv', index=False)
    keys = [key for key in ['theta', 'max_water_capacity', 'water_cell_density', 'feedback_rule'] if key in results]
    if keys:
        summary = results.groupby(keys, dropna=False).agg(
            fraction_low_variance_coop=('coop_stationary_low_variance', 'mean'),
            fraction_low_variance_env=('env_stationary_low_variance', 'mean'),
            median_tail_coop=('tail_mean_coop', 'median'),
            median_tail_env=('tail_mean_env', 'median'), n_runs=('RunId', 'count'),
        ).reset_index()
        summary.to_csv(args.output_dir / 'low_variance_summary.csv', index=False)
        fig, ax = plt.subplots(figsize=(7, 5))
        points = ax.scatter(summary.median_tail_env, summary.median_tail_coop,
                            c=summary.fraction_low_variance_coop, vmin=0, vmax=1)
        fig.colorbar(points, ax=ax, label='Fraction meeting descriptive tail criterion')
        ax.set(xlabel='Median tail environment', ylabel='Median tail cooperation',
               title='Low-variance tail diagnostics')
        fig.tight_layout()
        fig.savefig(args.output_dir / 'low_variance_summary.png', dpi=200)
        plt.close(fig)
    write_manifest(args.output_dir / 'low_variance_manifest.json',
                   {'analysis': 'per-run-low-variance-v2', 'tail_fraction': args.tail_fraction}, input_path=args.input)


if __name__ == '__main__':
    main()
