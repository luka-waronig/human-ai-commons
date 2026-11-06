"""Generate tables and figures from recorded artifacts, without simulations."""
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import Depletion, MetricSpec, joint_welfare, paired_interval, rescore, supported_activity
from .provenance import read_jsonl, rng, sha256, write_json
from .world import World
from .payoffs import Policy
from .diagnostics import spatial_summary


def verify_bundle(output):
    output = Path(output)
    checksums = json.loads((output/'checksums.json').read_text())
    for relative, expected in checksums.items():
        path = (output/relative).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'artifact checksum mismatch: {relative}')
    return dict(verified=True, files=len(checksums))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def analyze(output, destination=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    verify_bundle(output)
    destination = Path(destination) if destination else output/'analysis'
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((output/'manifest.json').read_text())
    label = manifest['configuration'].get('execution', {}).get('label', 'research')
    title_label = 'Software check (not study results)' if label.startswith('smoke') else label.replace('_', ' ')
    rows = read_jsonl(output/'realizations.jsonl')
    summaries = []
    for row in rows:
        summaries.append({key: value for key, value in row.items() if isinstance(value, (str, int, float, bool)) or value is None})
    write_csv(destination/'realization_metrics.csv', summaries)
    candidates = read_jsonl(output/'training_candidates.jsonl')
    coordinates = []
    for row in candidates:
        policy = Policy(**row['policy'])
        coordinates.append(dict(policy_id=policy.id, pressure=row['pressure'], u=policy.u, v=policy.v,
                                t=policy.t, s=policy.s, relative_c=policy.relative[0], relative_d=policy.relative[1],
                                q_c=policy.q_c, q_d=policy.q_d, slope_ratio=policy.omega, regime=policy.regime))
    write_csv(destination/'payoff_coordinates.csv', coordinates)
    joint = 'welfare' in manifest['configuration']['objective']
    measure = 'mean_welfare' if joint else 'mean_loss'
    weight = manifest['configuration']['objective'].get('welfare', {}).get('human_weight', .5)
    outcome_label = ('Human activity support' if weight == 1 else 'AI activity support' if weight == 0 else 'Joint activity support') if joint else 'Spatial-tail depletion loss'
    if candidates:
        loads = sorted(set(row['pressure'] for row in candidates))
        columns = min(2, len(loads))
        row_count = int(np.ceil(len(loads)/columns))
        fig, axes = plt.subplots(row_count, columns, figsize=(6*columns, 5*row_count), squeeze=False, constrained_layout=True)
        for ax in axes.flat[len(loads):]:
            ax.set_visible(False)
        for ax, pressure in zip(axes.flat, loads):
            points = [r for r in candidates if r['pressure'] == pressure and r['policy']['observation'] == 'local']
            scatter = ax.scatter([r['policy']['v'] for r in points], [r['policy']['u'] for r in points],
                                 c=[r[measure] for r in points], vmin=0, vmax=1, cmap='viridis', s=35)
            rejected = [r for r in points if not r.get('search_feasible', True)]
            if rejected:
                ax.scatter([r['policy']['v'] for r in rejected], [r['policy']['u'] for r in rejected],
                           marker='x', color='red', s=22, label='Persistence condition violated')
                ax.legend(fontsize=7)
            p = manifest['configuration']['payoffs']
            limits = np.array(p['relative_bounds'])
            ax.plot(-p['s']*limits, (p['t']-1)*limits, color='white', linewidth=.8)
            ax.set(xlabel='v: payoff change against a defector', ylabel='u: payoff change against a cooperator', title=f'Training outcomes; pressure={pressure}')
            fig.colorbar(scatter, ax=ax, label=outcome_label)
        fig.suptitle(f'{title_label}\nEvaluated depleted-state payoff changes', fontsize=11)
        fig.savefig(destination/'payoff_candidates.png', dpi=150)
        plt.close(fig)
    result_path = output/'test_comparisons.json'
    if result_path.exists():
        data = json.loads(result_path.read_text())
        table = []
        for problem in data['problems']:
            for family, row in problem['policies'].items():
                table.append(problem['problem'] | dict(family=family, mean_loss=row['mean_loss'],
                                                       policy_id=row['policy_id'], realizations=row['count']) |
                             {key: row[key] for key in ['mean_welfare', 'mean_human_service', 'mean_ai_service',
                              'stress_probability', 'stress_probability_upper', 'search_feasible',
                              'simultaneous_stress_probability_upper', 'simultaneous_protection_supported'] if key in row})
        write_csv(destination/'heldout_frontier.csv', table)
        write_csv(destination/'paired_comparisons.csv', [row['problem'] | row.get('welfare_local_vs_resource_only', row['local_vs_resource_only']) for row in data['problems']])
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for pressure in sorted(set(r['pressure'] for r in table)):
            for family in ['local', 'resource_only', 'equal_relative', 'redistribution']:
                group = sorted((r for r in table if r['pressure'] == pressure and r['family'] == family and r['norm'] == 'L2'), key=lambda r: r['budget'])
                if group:
                    ax.plot([r['budget'] for r in group], [r[measure] for r in group], marker='o',
                            label=f'{family}; pressure={pressure}')
                    if joint:
                        unsupported = [r for r in group if not r['simultaneous_protection_supported']]
                        ax.scatter([r['budget'] for r in unsupported], [r[measure] for r in unsupported],
                                   marker='x', color='red', zorder=3)
        ax.set(xlabel='Payoff-change budget b', ylabel=outcome_label,
               title=f'{title_label}\nSelected designs on independent test worlds')
        if joint:
            ax.scatter([], [], marker='x', color='red', label='Persistence requirement not established')
        ax.legend(fontsize=8)
        fig.savefig(destination/'heldout_frontier.png', dpi=150)
        plt.close(fig)
        primary = data['primary_problem']
        selected = next(r for r in data['problems'] if r['problem'] == primary)
        controls = {f'same matrix, {name}': row for name, row in data['same_matrix_observation_controls'].items()}
        controls.update({f'optimized {name}': row for name, row in selected['policies'].items() if name != 'local'})
        names, means, lower, upper = [], [], [], []
        for name, row in controls.items():
            values = np.asarray(row['welfares'] if joint else row['losses'])
            # An interval for a mean is the same bootstrap construction as a
            # paired reduction against zero, resampling whole realizations.
            interval = paired_interval(values, rng('control-mean', manifest['run_id'], name))
            names.append(name)
            means.append(values.mean())
            lower.append(max(0, values.mean()-interval['lower']))
            upper.append(max(0, interval['upper']-values.mean()))
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.barh(names, means, xerr=[lower, upper], color='#0369a1', capsize=3)
        ax.set(xlabel=f'{outcome_label}; 95% bootstrap interval',
               title=f'{title_label}\nObservation controls at the primary setting')
        fig.savefig(destination/'observation_controls.png', dpi=150)
        plt.close(fig)

        records = {r['evaluation_id']: r for r in rows}
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
        floor = manifest['configuration'].get('depletion_diagnostic', manifest['configuration']['objective'])['resource_floor']
        for ax, family in zip(axes, ['local', 'resource_only']):
            first = records[selected['policies'][family]['realization_ids'][0]]
            world = World.load(output/first['world_path'])
            with np.load(output/first['trajectory_path'], allow_pickle=False) as trajectory:
                deficit = np.maximum(floor-trajectory['stocks'][first['burn']:]/world.capacity, 0).mean(axis=0)/floor
                accessible = trajectory['accessible']
            image = np.full((world.spec.height, world.spec.width), np.nan)
            points = world.resources[accessible]
            image[points[:, 1], points[:, 0]] = deficit[accessible]
            plotted = ax.imshow(image, origin='lower', vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            ax.set(title=family, xlabel='Grid x', ylabel='Grid y')
            fig.colorbar(plotted, ax=ax, label='Cumulative normalized deficit')
        fig.suptitle(f'{title_label}: first held-out world, selected by index')
        fig.savefig(destination/'spatial_depletion.png', dpi=150)
        plt.close(fig)
        if joint:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
            for ax, family in zip(axes, ['local', 'resource_only']):
                first = records[selected['policies'][family]['realization_ids'][0]]
                world = World.load(output/first['world_path'])
                image = np.full((world.spec.height, world.spec.width), np.nan)
                points = world.resources[world.access.any(axis=0)] if first['stress_basis'] == 'resource_stock' else world.positions
                values = np.minimum(1, np.array(first['site_longest_continuous_stress_steps'])/first['stress_duration_steps'])
                for point, value in zip(points, values):
                    x, y = point
                    image[y, x] = value if np.isnan(image[y, x]) else max(image[y, x], value)
                plotted = ax.imshow(image, origin='lower', vmin=0, vmax=1, cmap='magma', interpolation='nearest')
                ax.set(title=family, xlabel='Grid x', ylabel='Grid y')
                fig.colorbar(plotted, ax=ax, label='Continuous stress duration / limit (capped at 1)')
            basis_label = 'Resource-stock persistence' if first['stress_basis'] == 'resource_stock' else 'Unmet-request persistence'
            fig.suptitle(f'{title_label}\n{basis_label}, first held-out world')
            fig.savefig(destination/'spatial_stress.png', dpi=150)
            plt.close(fig)
    capacity_rows = read_jsonl(output/'capacity_bounds.jsonl')
    write_csv(destination/'capacity_obstructions.csv', [
        {key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in capacity_rows])
    robustness_path = output/'robustness.json'
    if robustness_path.exists():
        robustness_rows = json.loads(robustness_path.read_text())
        table = []
        for row in robustness_rows:
            local, reference = row['policies']['local'], row['policies']['resource_only']
            if local['world_ids'] != reference['world_ids']:
                raise ValueError('robustness comparison has unpaired worlds')
            differences = (np.array(local['welfares'])-np.array(reference['welfares']) if joint
                           else np.array(reference['losses'])-np.array(local['losses']))
            interval = paired_interval(differences,
                                       rng('robustness-analysis', manifest['run_id'], row['variant']['name']))
            table.append(dict(variant=row['variant']['name'], **interval))
        write_csv(destination/'robustness_comparisons.csv', table)
        fig, ax = plt.subplots(figsize=(9, max(3, .3*len(table)+1)), constrained_layout=True)
        means = np.array([r['mean_reduction'] for r in table])
        errors = [np.maximum(0, means-[r['lower'] for r in table]), np.maximum(0, [r['upper'] for r in table]-means)]
        ax.errorbar(means, [r['variant'] for r in table], xerr=errors, fmt='o', capsize=3)
        ax.axvline(0, color='#6b7280', linestyle='--', linewidth=1)
        ax.set(xlabel=('Local minus resource-only welfare' if joint else 'Resource-only minus local loss')+'; 95% paired interval',
               title=f'{title_label}\nTransfer of frozen policies')
        fig.savefig(destination/'robustness.png', dpi=150)
        plt.close(fig)
    if joint:
        outcome_sensitivity(output, destination, manifest['configuration'])
    write_json(destination/'analysis_manifest.json', dict(
        run_id=manifest['run_id'], label=label, analysis_source_sha256=sha256(Path(__file__)),
        inference='Finite-horizon simulation results; smoke outputs are software checks.',
        input_sha256={name: sha256(output/name) for name in ['manifest.json', 'reference_characterization.json', 'training_candidates.jsonl', 'realizations.jsonl', 'test_comparisons.json'] if (output/name).exists()}))
    return dict(output=str(destination), rows=len(summaries), label=label)


def outcome_sensitivity(output, destination, config):
    """Re-score frozen policies; these tables do not establish new optima."""
    welfare_rows, stress_rows, pattern_rows = [], [], []
    records = {r['evaluation_id']: r for r in read_jsonl(Path(output)/'realizations.jsonl')}
    checks = config['secondary_checks']
    for row in records.values():
        if row['split'] not in ['test', 'characterization'] or 'trajectory_path' not in row:
            continue
        world = World.load(Path(output)/row['world_path'])
        with np.load(Path(output)/row['trajectory_path'], allow_pickle=False) as data:
            start = row['burn']
            pattern_rows.append(dict(evaluation_id=row['evaluation_id'], policy_id=row['policy_id'], split=row['split'],
                **spatial_summary(world, data['stocks'][start:], data['strategies'][start:],
                                  data['agent_requested'][start:], data['agent_delivered'][start:])))
            targets = data['agent_reference_request']*row['scenario']['defect_ratio']
            kinds = data['agent_is_ai']
            for extra_value in checks['additional_use_value']:
                support = supported_activity(data['agent_delivered'][row['burn']:], data['agent_reference_request'], targets, extra_value)
                for utility in checks['delivery_utility']:
                    value = support if utility == 'linear' else np.log1p(support)/np.log(2)
                    human, ai = float(value[:, ~kinds].mean()), float(value[:, kinds].mean())
                    for weight in checks['welfare_human_weight']:
                        welfare_rows.append(dict(evaluation_id=row['evaluation_id'], policy_id=row['policy_id'],
                            utility=utility, additional_use_value=extra_value, human_weight=weight, human_service=human, ai_service=ai,
                            welfare=float(joint_welfare(human, ai, weight)), scope='frozen_policy_rescoring'))
            normalized = data['stocks'][row['burn']:, data['accessible']]/world.capacity[data['accessible']]
            for reference in checks['stress_reference']:
                tracker = Depletion(normalized.shape[1], MetricSpec(reference, 1))
                for stocks in normalized:
                    tracker.update(stocks)
                for duration in checks['stress_duration_steps']:
                    if duration > len(normalized):
                        continue
                    fraction = float((tracker.longest_continuous_depletion_steps >= duration).mean())
                    stress_rows.append(dict(evaluation_id=row['evaluation_id'], policy_id=row['policy_id'],
                        basis='resource_stock', reference=reference, duration_steps=duration, persistent_cell_fraction=fraction,
                        persistent_local_stress=fraction > config['objective']['persistent_local_stress']['allowed_site_fraction'],
                        scope='frozen_policy_rescoring'))
    write_csv(Path(destination)/'welfare_assumptions.csv', welfare_rows)
    write_csv(Path(destination)/'stress_definitions.csv', stress_rows)
    write_csv(Path(destination)/'spatial_characterization.csv', pattern_rows)


def metric_sensitivity(output, floor, tail, destination):
    output = Path(output)
    verify_bundle(output)
    metric = MetricSpec(floor, tail)
    records = {r['evaluation_id']: r for r in read_jsonl(output/'realizations.jsonl')}
    rows = []
    for row in records.values():
        if row['split'] != 'test' or 'trajectory_path' not in row:
            continue
        world = World.load(output/row['world_path'])
        with np.load(output/row['trajectory_path'], allow_pickle=False) as data:
            metrics = rescore(data['stocks'], world.capacity, data['accessible'], row['burn'], metric)
        rows.append(dict(evaluation_id=row['evaluation_id'], policy_id=row['policy_id'], world_id=world.id,
                         pressure=row['scenario']['pressure'], floor=floor, tail=tail, **metrics))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_csv(destination, rows)
    return dict(output=str(destination), rescored_realizations=len(rows), simulations_started=False)
