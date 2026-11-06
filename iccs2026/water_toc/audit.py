"""Paired pilot comparison of identified replenishment variants.

This is a new sensitivity experiment, not a reproduction of published figures.
python -m water_toc.audit --replications 12 --steps 100
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from water_toc.feedback import FEEDBACK_RULES
from water_toc.model import WaterToC
from water_toc.reproducibility import stable_seed, write_manifest

CASES = [
    dict(theta=1.4, max_water_capacity=10, water_cell_density=.2),
    dict(theta=2., max_water_capacity=20, water_cell_density=.3),
    dict(theta=5., max_water_capacity=20, water_cell_density=.4),
    dict(theta=10., max_water_capacity=40, water_cell_density=.5),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replications', type=int, default=12)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--output-dir', type=Path, default=Path('results/reproducibility_audit'))
    args = parser.parse_args()
    if args.replications < 2 or args.steps < 10:
        parser.error('at least two replications and ten steps are required')
    rows = []
    for case_id, case in enumerate(CASES):
        for replicate in range(args.replications):
            seed = stable_seed('feedback-audit', replicate)
            for rule in FEEDBACK_RULES:
                model = WaterToC(**case, feedback_rule=rule, seed=seed,
                                 save_snapshots=False, record_spatial=False)
                for _ in range(args.steps):
                    model.step()
                frame = model.datacollector.get_model_vars_dataframe().iloc[1:]
                tail = frame.iloc[int(.8 * len(frame)):]
                rows.append(dict(case_id=case_id, replicate=replicate, seed=seed,
                    feedback_rule=rule, **case,
                    final_cooperation=float(frame.Coop_Fraction.iloc[-1]),
                    tail_cooperation=float(tail.Coop_Fraction.mean()),
                    tail_environment=float(tail.Environment_State.mean()),
                    delivery_fraction=float(frame.Delivered_Water.sum() / frame.Requested_Water.sum())))
        print(f'Finished scenario {case_id + 1}/{len(CASES)}', flush=True)
    runs = pd.DataFrame(rows)
    differences = []
    for case_id, group in runs.groupby('case_id'):
        reference = group[group.feedback_rule == 'repository'].set_index('replicate')
        for rule in FEEDBACK_RULES[1:]:
            alternative = group[group.feedback_rule == rule].set_index('replicate')
            for metric in ['tail_cooperation','tail_environment','delivery_fraction']:
                difference = alternative[metric] - reference[metric]
                differences.append(dict(case_id=case_id, alternative=rule, metric=metric,
                    n_pairs=len(difference), mean_paired_difference=float(difference.mean()),
                    paired_difference_sd=float(difference.std(ddof=1))))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.output_dir / 'feedback_variant_runs.csv', index=False)
    pd.DataFrame(differences).to_csv(args.output_dir / 'feedback_variant_differences.csv', index=False)
    write_manifest(args.output_dir / 'manifest.json',
        dict(experiment='paired replenishment sensitivity pilot', cases=CASES,
             replications=args.replications, steps=args.steps, seed_scheme='sha256 feedback-audit/replicate',
             publication_reproduction=False), input_path=args.output_dir / 'feedback_variant_runs.csv')
    print(pd.DataFrame(differences).to_string(index=False))


if __name__ == '__main__':
    main()
