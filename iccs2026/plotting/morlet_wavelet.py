"""Centered per-run Morlet power, with edge exclusion and at least four cycles.

python -m plotting.morlet_wavelet --input <sweep.csv>
Power is descriptive; this script supplies no wavelet significance test.
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
from water_toc.reproducibility import run_identifier, write_manifest


def compute_cwt_power(x, t, min_period=5., max_period=None, n_freqs=64):
    x, t = np.asarray(x, float), np.asarray(t, float)
    if x.ndim != 1 or x.shape != t.shape or len(x) < 8 or not np.isfinite(x).all() or not np.isfinite(t).all():
        raise ValueError('a finite time series of at least eight samples is required')
    intervals = np.diff(t)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0]):
        raise ValueError('times must be regularly spaced and increasing')
    dt = intervals[0]
    supported_max = (t[-1] - t[0]) / 4
    max_period = supported_max if max_period is None else min(max_period, supported_max)
    if not 2 * dt <= min_period < max_period:
        raise ValueError('time window cannot resolve the requested period range')
    frequencies = np.linspace(1 / max_period, 1 / min_period, n_freqs)
    scales = pywt.central_frequency('morl') / (frequencies * dt)
    coefficients, _ = pywt.cwt(x - np.mean(x), scales, 'morl', sampling_period=dt)
    power = np.abs(coefficients) ** 2
    # Approximate Morlet e-folding edge exclusion; not a significance criterion.
    edge_distance = np.minimum(np.arange(len(x)), np.arange(len(x))[::-1])
    power[edge_distance[None, :] < np.sqrt(2) * scales[:, None]] = np.nan
    return power, dt, scales


def oscillation_score(power, top_pct=90.):
    """Mean power at or above the chosen percentile, excluding masked edges."""
    finite = np.asarray(power)[np.isfinite(power)]
    if not len(finite):
        return np.nan
    threshold = np.percentile(finite, top_pct)
    return float(finite[finite >= threshold].mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=Path('water_toc_sweep_results.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('results/corrected_analysis/wavelets'))
    args = parser.parse_args()
    frame = pd.read_csv(args.input).rename(columns={'Step':'step'})
    required = {'RunId','step','Coop_Fraction'}
    if not required.issubset(frame):
        raise ValueError(f'Missing columns: {sorted(required - set(frame))}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records, panels = [], {}
    for run_id, group in frame.groupby('RunId', sort=False):
        group = group[group.step > 0].sort_values('step')
        group = group.iloc[int(.1 * len(group)):]
        t, x = group.step.to_numpy(), group.Coop_Fraction.to_numpy()
        metadata = {key:group[key].iloc[0] for key in
                    ['theta','max_water_capacity','water_cell_density','feedback_rule'] if key in group}
        try:
            power, dt, scales = compute_cwt_power(x,t)
        except ValueError as error:
            records.append({'RunId':run_id, **metadata, 'status':str(error), 'score':np.nan})
            continue
        records.append({'RunId':run_id, **metadata, 'status':'computed', 'score':oscillation_score(power)})
        key = run_identifier(metadata)
        panels.setdefault(key, []).append((t,power,dt,scales,metadata))
    pd.DataFrame(records).to_csv(args.output_dir / 'per_run_wavelet_scores.csv', index=False)
    for key, runs in panels.items():
        t,power,dt,scales,metadata = runs[0]
        if any(not np.array_equal(item[0],t) for item in runs):
            continue  # Per-run scores remain valid; unequal time grids are not averaged.
        stack = np.stack([item[1] for item in runs])
        count = np.isfinite(stack).sum(axis=0)
        mean = np.divide(np.nansum(stack,axis=0),count,out=np.full_like(power,np.nan),where=count>0)
        frequencies = pywt.central_frequency('morl') / (scales * dt)
        fig,ax = plt.subplots(figsize=(7,4))
        picture = ax.imshow(mean,origin='lower',aspect='auto',
            extent=[t[0],t[-1],frequencies[0],frequencies[-1]],cmap='inferno')
        description = ', '.join(f'{label}={metadata[key]}' for key,label in
            [('theta','theta'),('max_water_capacity','K'),('water_cell_density','density'),('feedback_rule','rule')]
            if key in metadata)
        ax.set(xlabel='Step',ylabel='Frequency (1/step)',title=f'Mean centered per-run Morlet power\n{description}')
        fig.colorbar(picture,ax=ax,label='Squared Morlet coefficient')
        fig.tight_layout()
        fig.savefig(args.output_dir / f'wavelet_{key}.png',dpi=200)
        plt.close(fig)
    write_manifest(args.output_dir / 'manifest.json',
        {'analysis':'centered-per-run-power-v2','minimum_cycles':4,'edge_rule':'sqrt(2)*scale'}, input_path=args.input)


if __name__ == '__main__':
    main()
