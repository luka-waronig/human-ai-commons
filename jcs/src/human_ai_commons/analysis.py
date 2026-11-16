"""Tables and figures from recorded artifacts, without simulations."""
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import Depletion, MetricSpec, joint_welfare, paired_interval, rescore, supported_activity
from .provenance import read_jsonl, rng, sha256, write_json
from .world import World
from .payoffs import Policy
from .diagnostics import spatial_summary


def verify_bundle(output, progress=None):
    output = Path(output)
    checksums = json.loads((output/'checksums.json').read_text())
    for completed, (relative, expected) in enumerate(checksums.items(), 1):
        path = (output/relative).resolve()
        if not path.is_relative_to(output.resolve()) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f'artifact checksum mismatch: {relative}')
        if progress is not None and (completed % 2000 == 0 or completed == len(checksums)):
            progress(completed, len(checksums))
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
    from .frontiers import enabled
    if enabled(manifest['configuration']):
        return analyze_frontiers(output, destination, manifest, rows, candidates, title_label)
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
            # an interval for a mean is the same bootstrap construction as a
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


def analyze_frontiers(output, destination, manifest, records, training, title):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from .frontiers import nondominated
    write_csv(destination/'training_pareto_candidates.csv', [dict(
        pressure=r['pressure'], policy_id=r['policy_id'], observation=r['policy']['observation'],
        u=r['policy']['u'], v=r['policy']['v'], human=r['mean_human_service'], ai=r['mean_ai_service'],
        depletion=r['mean_loss'], empirical_constraint_satisfied=r['search_feasible']) for r in training])
    result_path = output/'test_comparisons.json'
    table = []
    if result_path.exists():
        data = json.loads(result_path.read_text())
        for problem in data['problems']:
            reference = problem['families']['reference'][0]
            for family, candidates in problem['families'].items():
                if family == 'reference':
                    continue
                alternatives = list({r['policy_id']: r for r in candidates+[reference]}.values())
                empirical = {r['policy_id'] for r in nondominated([r for r in alternatives if r['search_feasible']], deduplicate=False)}
                supported = {r['policy_id'] for r in nondominated([r for r in alternatives if r['simultaneous_protection_supported']], deduplicate=False)}
                for candidate in alternatives:
                    policy = Policy(**candidate['policy'])
                    row = dict(problem_id=problem['problem_id'], **problem['problem'], family=family, policy_id=policy.id,
                        is_reference=policy.id == reference['policy_id'],
                        u=policy.u, v=policy.v, q_c=policy.q_c, q_d=policy.q_d, payoff_cost=policy.cost(problem['problem']['norm']),
                        human=candidate['mean_human_service'], ai=candidate['mean_ai_service'], depletion=candidate['mean_loss'],
                        realizations=candidate['count'], stress_probability=candidate['stress_probability'],
                        simultaneous_stress_probability_upper=candidate['simultaneous_stress_probability_upper'],
                        resource_family_confidence=candidate['family_confidence'],
                        protected_family_size=candidate['protected_family_size'],
                        empirical_constraint_satisfied=candidate['search_feasible'],
                        constraint_supported=candidate['simultaneous_protection_supported'],
                        estimated_family_pareto=policy.id in empirical,
                        supported_family_pareto=policy.id in supported,
                        frontier_scope='sample means among frozen tested designs and original game; no global or statistical dominance claim')
                    for name, field in [('human', 'human_services'), ('ai', 'ai_services'), ('depletion', 'losses')]:
                        interval = paired_interval(candidate[field], rng('pareto-mean', problem['problem_id'], policy.id, name))
                        row[name+'_lower'], row[name+'_upper'] = interval['lower'], interval['upper']
                    table.append(row)
        write_csv(destination/'pareto_candidates.csv', table)
        # headers are retained even when no design establishes the constraint.
        for name, flag in [('pareto_frontier.csv', 'supported_family_pareto'),
                           ('estimated_pareto_frontier.csv', 'estimated_family_pareto')]:
            with (destination/name).open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(table[0]))
                writer.writeheader()
                writer.writerows(r for r in table if r[flag])
        write_csv(destination/'paired_outcome_comparisons.csv', data['paired_outcome_comparisons'])
        ids = list(dict.fromkeys(r['problem_id'] for r in table))
        columns = min(2, len(ids))
        fig, axes = plt.subplots(int(np.ceil(len(ids)/columns)), columns,
            figsize=(7*columns, 5*int(np.ceil(len(ids)/columns))), squeeze=False, constrained_layout=True)
        colors = plt.get_cmap('viridis')
        markers = {'local': 'o', 'resource_only': 's', 'equal_relative': 'D', 'redistribution': '^', 'global': 'P'}
        for ax in axes.flat[len(ids):]:
            ax.set_visible(False)
        for ax, pid in zip(axes.flat, ids):
            group = [r for r in table if r['problem_id'] == pid]
            for row in group:
                if row['is_reference']:
                    ax.scatter(row['human'], row['ai'], marker='*', s=150, color='black', zorder=5)
                else:
                    marker = markers[row['family']] if row['empirical_constraint_satisfied'] else 'x'
                    color = colors(row['depletion'])
                    if marker == 'x':
                        ax.scatter(row['human'], row['ai'], marker=marker, color=[color], s=45, alpha=.7)
                    else:
                        ax.scatter(row['human'], row['ai'], marker=marker, s=65,
                            facecolors=[color] if row['constraint_supported'] else 'none',
                            edgecolors='black' if row['supported_family_pareto'] else [color],
                            linewidths=1.2, zorder=3 if row['estimated_family_pareto'] else 2)
            first = group[0]
            ax.set(xlabel='Human activity supported', ylabel='AI activity supported', xlim=(0, 1.02), ylim=(0, 1.02),
                   title=f"Pressure {first['pressure']}; budget {first['budget']}; {first['norm']}")
            ax.grid(alpha=.15)
        handles = [Line2D([], [], marker=m, linestyle='', color='#555555', label=f.replace('_', ' ')) for f, m in markers.items()]
        handles += [Line2D([], [], marker='*', linestyle='', color='black', label='original game'),
            Line2D([], [], marker='o', linestyle='', markerfacecolor='none', color='#555555', label='constraint not established'),
            Line2D([], [], marker='x', linestyle='', color='#555555', label='empirical constraint violated')]
        fig.legend(handles=handles, loc='outside lower center', ncol=3, fontsize=8)
        fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=colors), ax=list(axes.flat[:len(ids)]),
                     label='Local depletion (lower is better)', shrink=.75)
        fig.suptitle(title+'\nThree-objective Pareto estimates projected onto human and AI support\n'
            'Filled: resource constraint established; black edge: estimated family frontier within that set', fontsize=11)
        for suffix in ['png', 'pdf']:
            fig.savefig(destination/f'pareto_frontier.{suffix}', dpi=180)
        plt.close(fig)
        write_json(destination/'pareto_summary.json', dict(
            problems=[dict(problem_id=pid,
                tested_designs=len({r['policy_id'] for r in table if r['problem_id'] == pid}),
                constraint_supported_designs=len({r['policy_id'] for r in table if r['problem_id'] == pid and r['constraint_supported']}),
                status='constraint_supported_designs_found' if any(r['constraint_supported'] for r in table if r['problem_id'] == pid)
                       else 'no_design_established_the_constraint') for pid in ids],
            note='Empty supported frontiers do not prove infeasibility. Bootstrap mean intervals are descriptive; dominance uses sample means.'))
        frontier_spatial_figures(output, destination, data, records, title)
        write_csv(destination/'observation_controls.csv', [dict(
            source_policy_id=c['policy_id'], observation=observation, policy_id=r['policy_id'],
            human=r['mean_human_service'], ai=r['mean_ai_service'], depletion=r['mean_loss'],
            scope='same matrix with changed observations, not a reoptimized frontier')
            for c in data['same_matrix_observation_controls'] for observation, r in c['observations'].items()])
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        for observation, marker in [('local', 'o'), ('global', 'P'), ('fixed', 's')]:
            points = [r['observations'][observation] for r in data['same_matrix_observation_controls']]
            plotted = ax.scatter([r['mean_human_service'] for r in points], [r['mean_ai_service'] for r in points],
                c=[r['mean_loss'] for r in points], vmin=0, vmax=1, cmap='viridis', marker=marker, label=observation)
        ax.set(xlabel='Human activity supported', ylabel='AI activity supported', xlim=(0, 1.02), ylim=(0, 1.02),
               title=title+'\nFixed matrices under different observations')
        ax.legend()
        fig.colorbar(plotted, ax=ax, label='Local depletion')
        fig.savefig(destination/'observation_controls.png', dpi=150)
        plt.close(fig)
    robustness_path = output/'robustness.json'
    if robustness_path.exists():
        transferred = json.loads(robustness_path.read_text())
        write_csv(destination/'robustness_comparisons.csv', [dict(variant=v['variant']['name'], family=family,
            policy_id=c['policy_id'], human=c['mean_human_service'], ai=c['mean_ai_service'], depletion=c['mean_loss'],
            realizations=c['count'], stress_probability=c['stress_probability'],
            simultaneous_stress_probability_upper=c['simultaneous_stress_probability_upper'],
            resource_family_confidence=c['family_confidence'], protected_family_size=c['protected_family_size'],
            empirical_constraint_satisfied=c['search_feasible'], constraint_supported=c['simultaneous_protection_supported'],
            scope='frozen-set transfer, not a reoptimized frontier')
            for v in transferred for family, candidates in v['families'].items() for c in candidates])
        write_csv(destination/'robustness_outcome_comparisons.csv',
                  [r for v in transferred for r in v['paired_outcome_comparisons']])
        if transferred:
            fig, axes = plt.subplots(1, 3, figsize=(14, max(3, .35*len(transferred))), constrained_layout=True)
            for ax, field, label in zip(axes, ['mean_human_service', 'mean_ai_service', 'mean_loss'],
                                       ['Human support change', 'AI support change', 'Depletion change']):
                for index, variant in enumerate(transferred):
                    reference = variant['families']['reference'][0][field]
                    for family, candidates in variant['families'].items():
                        if family != 'reference':
                            ax.scatter([c[field]-reference for c in candidates], [index]*len(candidates),
                                       marker={'local': 'o', 'resource_only': 's', 'equal_relative': 'D',
                                               'redistribution': '^', 'global': 'P'}[family], s=20, alpha=.6)
                ax.axvline(0, color='black', linewidth=.7)
                ax.set(yticks=range(len(transferred)), yticklabels=[v['variant']['name'] for v in transferred], xlabel=label)
            fig.suptitle(title+'\nFrozen-design changes from the original game; not reoptimized frontiers')
            fig.savefig(destination/'robustness.png', dpi=150)
            plt.close(fig)
    outcome_sensitivity(output, destination, manifest['configuration'])
    write_json(destination/'analysis_manifest.json', dict(run_id=manifest['run_id'],
        analysis_source_sha256=sha256(Path(__file__)), label=manifest['configuration'].get('execution', {}).get('label', 'research'),
        inference='Estimated finite-horizon Pareto sets; software-check results are not research evidence.',
        input_sha256={name: sha256(output/name) for name in ['manifest.json', 'reference_characterization.json',
            'training_candidates.jsonl', 'realizations.jsonl', 'shortlists.json', 'frozen_policies.json',
            'test_comparisons.json', 'robustness.json'] if (output/name).exists()}))
    return dict(output=str(destination), rows=len(records), tested_designs=len(table))


def frontier_spatial_figures(output, destination, data, records, title):
    """Matched spatial illustrations of the original game and frontier ends."""
    import matplotlib.pyplot as plt
    from .frontiers import spread
    primary = next(r for r in data['problems'] if r['problem'] == data['primary_problem'])
    selected = [('original game', primary['families']['reference'][0])]
    selected += [('payoff '+r['policy_id'][:6], r) for r in spread(primary['families']['local'], 3)]
    lookup = {r['evaluation_id']: r for r in records}
    for name, duration in [('spatial_depletion', False), ('spatial_stress', True)]:
        fig, axes = plt.subplots(1, len(selected), figsize=(4*len(selected), 4), squeeze=False, constrained_layout=True)
        for ax, (label, candidate) in zip(axes.flat, selected):
            record = lookup[candidate['realization_ids'][0]]
            world = World.load(output/record['world_path'])
            image = np.full((world.spec.height, world.spec.width), np.nan)
            accessible = world.access.any(axis=0)
            if duration:
                points = world.resources[accessible] if record['stress_basis'] == 'resource_stock' else world.positions
                values = np.minimum(1, np.array(record['site_longest_continuous_stress_steps'])/record['stress_duration_steps'])
            else:
                with np.load(output/record['trajectory_path'], allow_pickle=False) as trajectory:
                    floor = record['metric']['floor']
                    values = np.maximum(floor-trajectory['stocks'][record['burn']:, accessible]/world.capacity[accessible], 0).mean(axis=0)/floor
                points = world.resources[accessible]
            for (x, y), value in zip(points, values):
                image[y, x] = value if np.isnan(image[y, x]) else max(image[y, x], value)
            plotted = ax.imshow(image, origin='lower', vmin=0, vmax=1, cmap='magma', interpolation='nearest')
            ax.set(title=label, xlabel='Grid x', ylabel='Grid y')
            fig.colorbar(plotted, ax=ax, label='Stress duration / limit' if duration else 'Mean normalized resource deficit')
        fig.suptitle(title+'\nFirst independent test world, selected by index; illustrative spatial comparison', fontsize=10)
        fig.savefig(destination/(name+'.png'), dpi=150)
        plt.close(fig)


def outcome_sensitivity(output, destination, config):
    """Rescoring of frozen policies; these tables do not establish new optima."""
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


def relative_change_percent(values, reference):
    """Descriptive change relative to a positive reference, in percent."""
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('a finite positive reference is required')
    return 100 * (np.asarray(values, dtype=float) / reference - 1)


def combined_frontiers(candidates):
    """Combined three-objective sample frontiers within identical problem settings."""
    import pandas as pd
    rows = []
    for _, group in candidates.groupby('problem_id', sort=False):
        group = group.drop_duplicates('policy_id').copy()
        objectives = np.column_stack([-group.human, -group.ai, group.depletion])
        if not np.isfinite(objectives).all():
            raise ValueError('frontier outcomes must be finite')
        dominated = np.any(np.all(objectives[:, None, :] <= objectives[None, :, :], axis=2)
                           & np.any(objectives[:, None, :] < objectives[None, :, :], axis=2), axis=0)
        group['unconstrained_combined_sample_frontier'] = ~dominated
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def study_plots(bundle, derived, destination):
    """Study plots from archived outcomes and explicitly supplied derived data; no simulations."""
    import hashlib
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator
    bundle, D, destination = Path(bundle).resolve(), Path(derived).resolve(), Path(destination).resolve()
    if destination.is_relative_to(bundle) or destination == D or destination.is_relative_to(D):
        raise ValueError('Plot output must be separate from archived and derived inputs')
    manifest = json.loads((bundle/'manifest.json').read_text())
    provenance = json.loads((D/'provenance_and_verification.json').read_text())
    if provenance['run_id'] != manifest['run_id']:
        raise ValueError('Derived results belong to a different recorded run')
    destination.mkdir(parents=True, exist_ok=True)
    F = destination/'figures'
    F.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Serif','font.size':14,'axes.labelsize':14,'axes.titlesize':14,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    blue='#244867';gray='#777777';B='db0eed394e20c822b2a6beb3';A='8b24b6293b7838f8ae4cf9e4';O='de23924de8b26afc53a5dc84';E='fdb3c78ea72489620aec414d';N='9ca5ba4d95825c02d01d9888';G='73bea62e1e1318e4d8974bc6';Low='e9883'
    cand=pd.read_csv(bundle/'analysis/pareto_candidates.csv');f=combined_frontiers(cand);c=pd.read_csv(bundle/'analysis/paired_outcome_comparisons.csv').drop_duplicates(['pressure','policy_id','outcome']);m=pd.read_csv(D/'trajectory_condition_means.csv');m=m[m.split=='test'];means={(r.pressure,r.policy_id):r for _,r in m.iterrows()};bands=pd.read_csv(D/'all_one_action_responses_and_bands.csv')
    for row in cand.drop_duplicates(['pressure','policy_id']).itertuples():
        q=means.get((row.pressure,row.policy_id))
        if q is None or not np.allclose([q.late_human_support,q.late_ai_support],[row.human,row.ai],rtol=0,atol=1e-10):
            raise ValueError('Derived activity summary disagrees with archived test means: '+row.policy_id)
    labels={B:'Original',A:'A',O:'B',E:'C',N:'D',G:'G'}
    primary=f[(f.pressure==1)&(f.budget==.5)&(f.norm=='L2')].copy();others=sorted(set(primary.policy_id)-set(labels))
    labels.update({p:f'P{i+1:02d}' for i,p in enumerate(others)})
    (destination/'plot_labels.json').write_text(json.dumps(labels,indent=2))
    def save(fig,n):
     fig.savefig(F/(n+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None});fig.savefig(F/(n+'.png'),dpi=150,bbox_inches='tight');plt.close(fig)
    def coords(g):
     b=g[g.policy_id==B].iloc[0]
     return np.column_stack([relative_change_percent(g.human,b.human),relative_change_percent(g.ai,b.ai),relative_change_percent(g.depletion,b.depletion)])
    def dots(ax,g,x,y,annotate=False):
     v=coords(g)
     if y==2: v[:,2] *= -10000
     for flag in [False,True]:
      q=g.unconstrained_combined_sample_frontier.to_numpy()==flag
      ax.scatter(v[q,x],v[q,y],s=38,facecolors=blue if flag else 'white',edgecolors=blue if flag else gray,lw=.9,zorder=3)
     k=np.flatnonzero(g.policy_id.to_numpy()==B)[0];ax.scatter(v[k,x],v[k,y],marker='*',s=130,c='black',zorder=5)
     if annotate:
      for pid in [A,O,E,G]:
       z=np.flatnonzero(g.policy_id.to_numpy()==pid)
       if len(z):k=z[0];ax.annotate(labels[pid],(v[k,x],v[k,y]),xytext=(4,5 if pid!=O else -11),textcoords='offset points',fontsize=13)
     ax.axhline(0,color='#bbb',lw=.6);ax.axvline(0,color='#bbb',lw=.6);ax.grid(alpha=.1);ax.ticklabel_format(useOffset=False,style='plain');ax.xaxis.set_major_locator(MaxNLocator(4));ax.yaxis.set_major_locator(MaxNLocator(4))
    fig,axs=plt.subplots(1,3,figsize=(11,3.8),layout='constrained')
    for ax,(x,y,title) in zip(axs,[(0,1,'a  Human–AI activity trade-offs'),(0,2,'b  Human activity and depletion'),(1,2,'c  AI activity and depletion')]):
     dots(ax,primary,x,y,x==0 and y==1);names=['Human support gain\n(% of original)','AI support gain\n(% of original)','Depletion reduction\n(ppm of original)'];ax.set(xlabel=names[x],ylabel=names[y],title=title)
     if y==2:
            ax.set_yticks([0,.1,.2,.3]);ax.set_ylim(-.04,.3)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x,p:f'{x:.1f}'))
    fig.legend(handles=[Line2D([],[],marker='o',ls='',color=blue,label='On three-objective mean frontier'),Line2D([],[],marker='o',ls='',markerfacecolor='white',color=gray,label='Other tested design'),Line2D([],[],marker='*',ls='',color='black',ms=10,label='Original game')],loc='outside lower center',ncol=3,frameon=False,fontsize=14)
    save(fig,'primary_tradeoffs')
    primary.assign(human_relative_change=coords(primary)[:,0],ai_relative_change=coords(primary)[:,1],depletion_relative_change=coords(primary)[:,2],depletion_reduction_ppm=-10000*coords(primary)[:,2]).to_csv(destination/'primary_pareto_values.csv',index=False)
    # original simultaneous outcome intervals: examples span the actual principal settings and families.
    low=c[(c.pressure==.25)&(c.outcome=='depletion')].sort_values('difference').iloc[0].policy_id
    specs=[(1,A,'A: local participation'),(1,O,'B: offsetting changes'),(1,E,'C: earlier restraint'),(1,G,'G: global participation'),(1,N,'D: later restraint, budget 1'),(.25,low,'Lowest mean depletion, pressure 0.25')]
    (destination/'interval_examples.json').write_text(json.dumps(specs))
    fig,axs=plt.subplots(1,3,figsize=(10,4.6),layout='constrained')
    for ax,field,title in zip(axs,['human','ai','depletion'],['a  Human support','b  AI support','c  Depletion score']):
     for j,(p,pid,label) in enumerate(specs):
      q=c[(c.pressure==p)&(c.policy_id==pid)&(c.outcome==field)].iloc[0]
      ax.errorbar(q.difference*100,j,xerr=[[100*(q.difference-q.simultaneous_lower)],[100*(q.simultaneous_upper-q.difference)]],fmt='o',color=blue,capsize=2,ms=4)
     ax.axvline(0,color='black',lw=.8);ax.set(title=title,xlabel='Difference\n(percentage points)',yticks=range(len(specs)));ax.invert_yaxis();ax.grid(axis='y',alpha=.12)
    axs[0].set_yticklabels([s[2] for s in specs],fontsize=13)
    for ax in axs[1:]:ax.set_yticklabels([])
    save(fig,'paired_outcomes')
    pd.concat([c[(c.pressure==p)&(c.policy_id==pid)] for p,pid,_ in specs]).to_csv(destination/'planned_comparison_values.csv',index=False)
    # coverage ratios, within-group tails and affected-source fractions.
    b=m[m.policy_id==B].sort_values('pressure');p=b.pressure.to_numpy()
    fig,axs=plt.subplots(1,3,figsize=(10.5,3.7),layout='constrained')
    axs[0].plot(p,100*b.late_ai_support/b.late_human_support,'o-',color=blue);axs[0].set(title='a  Relative activity\ncoverage',ylabel='AI support / human support\n(%)',ylim=(0,105))
    for metric,den,label,col,ls in [('human_worst10_agent_support','late_human_support','Human',blue,'-'),('ai_worst10_agent_support','late_ai_support','AI',gray,'--')]:axs[1].plot(p,100*b[metric]/b[den],marker='o',color=col,ls=ls,label=label)
    axs[1].set(title='b  Within-group\ndisadvantage',ylabel='Worst tenth / group mean\n(%)',ylim=(0,105));axs[1].legend(frameon=False)
    axs[2].plot(p,100*b.persistent_cell_fraction,'o-',color=blue);axs[2].set(title='c  Breadth of\nprolonged low stocks',ylabel='Sources with prolonged\nlow stocks (%)',ylim=(0,105))
    for ax in axs:ax.set(xlabel='Cooperative demand /\nbaseline renewal',xticks=p,xticklabels=[f'{value:g}' for value in p]);ax.grid(alpha=.12)
    save(fig,'spatial_burdens')
    b.assign(ai_to_human_support_ratio=b.late_ai_support/b.late_human_support,human_tail_to_mean=b.human_worst10_agent_support/b.late_human_support,ai_tail_to_mean=b.ai_worst10_agent_support/b.late_ai_support).to_csv(destination/'spatial_burden_values.csv',index=False)
    # one-action stock and neighborhood-cooperation responses.
    def response(pid,stem):
     q=bands[(bands.pressure==1)&(bands.policy_id==pid)&(bands.comparison=='one_action_cooperate_minus_defect')&bands.metric.str.startswith(stem)].copy();q['h']=q.metric.str.rsplit('h',n=1).str[-1].astype(int);return q.sort_values('h')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.2),layout='constrained')
    for stem,label,col,ls in [('late_ipw_source_stock_h','At the source used',blue,'-'),('late_ipw_future_sensed_stock_h',"At the actor’s later source",gray,'--')]:
     q=response(B,stem);q=q[q.h>=1];axs[0].plot(q.h,q['mean']*100,marker='o',label=label,color=col,ls=ls);axs[0].fill_between(q.h,q.exploratory_joint_lower*100,q.exploratory_joint_upper*100,color=col,alpha=.12)
    axs[0].set(title='a  Resource benefits at\ndifferent sources',xlabel='Steps after one lower request',ylabel='Extra stock\n(percentage points of capacity)');axs[0].legend(frameon=False,fontsize=13)
    for pid,label,col,ls in [(B,'Original',gray,':'),(A,'Local A',blue,'-'),(O,'Offsetting B',blue,'--'),(E,'Earlier restraint C','black','-.')]:
     q=response(pid,'late_ipw_neighborhood_cooperation_h');q=q[q.h.isin([1,2,5,10])];axs[1].plot(q.h,q['mean']*100,marker='o',color=col,ls=ls,label=label);axs[1].fill_between(q.h,q.exploratory_joint_lower*100,q.exploratory_joint_upper*100,color=col,alpha=.07)
    axs[1].set(title='b  A brief change in\ncooperation feedback',xlabel='Steps after one lower request',ylabel='Nearby cooperation change\n(percentage points)');axs[1].legend(frameon=False,fontsize=13)
    axs[1].set_xticks([1,2,5,10])
    for ax in axs:ax.axhline(0,color='#999',lw=.6)
    save(fig,'action_responses')
    bands[(bands.pressure==1)&bands.policy_id.isin([B,A,O,E])].to_csv(destination/'feedback_values.csv',index=False)
    # matched-window activity and resource comparisons.
    profiles={g['policy_id']:g['profiles'] for g in json.loads((D/'complete_profiles.json').read_text()) if g['split']=='test' and g['pressure']==1}
    # first-500-step cooperation and stock paths at the same permitted payoff budget.
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), layout='constrained')
    formation_values = []
    for pid, color, style, label in [(A, blue, '-', 'Local participation A'),
                                      (O, gray, '--', 'Offsetting payoffs B'),
                                      (E, 'black', '-.', 'Earlier restraint C')]:
        cooperation = np.asarray(profiles[pid]['mean_cooperation'][:500])
        original_cooperation = np.asarray(profiles[B]['mean_cooperation'][:500])
        stock = np.asarray(profiles[pid]['mean_stock'][:500])
        original_stock = np.asarray(profiles[B]['mean_stock'][:500])
        steps = np.arange(1, 501)
        axes[0].plot(steps, 100*(cooperation-original_cooperation), color=color, ls=style, lw=1.8, label=label)
        axes[1].plot(steps, 100*(stock-original_stock), color=color, ls=style, lw=1.8, label=label)
        formation_values.extend(dict(policy_id=pid, step=int(t), cooperation_mean=float(c0),
                                     reference_cooperation_mean=float(c1), cooperation_change_pp=float(100*(c0-c1)),
                                     stock_mean=float(n0), reference_stock_mean=float(n1), stock_change_pp=float(100*(n0-n1)))
                                for t, c0, c1, n0, n1 in zip(steps, cooperation, original_cooperation, stock, original_stock))
    axes[0].set(title='a  Changes in lower requests', xlabel='Simulation step',
                ylabel='Cooperation change\n(percentage points)')
    axes[1].set(title='b  Extra water remaining', xlabel='Simulation step',
                ylabel='Stock change\n(percentage points of capacity)')
    for ax in axes:
        ax.axhline(0, color='#999999', lw=.7)
        ax.grid(axis='y', alpha=.12)
    fig.legend(handles=axes[0].get_lines()[:3], loc='outside lower center', ncol=3, frameon=False, fontsize=14)
    save(fig, 'early_behavior')
    pd.DataFrame(formation_values).to_csv(destination/'formation_equal_budget_values.csv', index=False)
    # cumulative renewal and withdrawal contributions to stock differences.
    budgets=pd.read_csv(D/'paired_resource_budget_means.csv')
    selected_budget=budgets[(budgets.split=='test')&(budgets.pressure==1)&budgets.policy_id.isin([E,N])]
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.8),layout='constrained')
    for ax,pid,title in zip(axs,[E,N],['a  Earlier restraint C','b  Later restraint D']):
        q=selected_budget[selected_budget.policy_id==pid].sort_values('step');x=np.arange(len(q))
        renewal=100*q.actual_renewal.to_numpy();saved=-100*q.withdrawal.to_numpy();net=100*q.stock.to_numpy()
        ax.bar(x-.17,renewal,width=.30,color=blue,label='Change in water renewed')
        ax.bar(x+.17,saved,width=.30,color=gray,label='Water retained through lower withdrawals')
        ax.scatter(x,net,color='black',marker='D',s=35,zorder=5,label='Net extra water remaining')
        ax.axhline(0,color='#999',lw=.7);ax.set(title=title,xticks=x,xticklabels=['100','500','2,000'],xlabel='Simulation step',ylabel='Water difference\n(% of total source capacity)');ax.grid(axis='y',alpha=.1)
    handles,legend_labels=axs[0].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='outside lower center',ncol=1,frameon=False,fontsize=12)
    save(fig,'resource_budgets');selected_budget.to_csv(destination/'resource_accounting_values.csv',index=False)
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.2),layout='constrained')
    profile_values=[];activity_values=[]
    for pid,col,ls,label in [(E,blue,'-','Earlier restraint C'),(N,gray,'--','Later restraint D')]:
     a=np.array(profiles[pid]['mean_stock'][:500]);base=np.array(profiles[B]['mean_stock'][:500]);axs[0].plot(np.arange(1,501),100*(a/base-1),color=col,ls=ls,label=label,lw=1.8)
     fields=['first100_human_support','first100_ai_support','late_human_support','late_ai_support'];vals=[100*(means[(1,pid)][f]/means[(1,B)][f]-1) for f in fields];y=np.arange(4)+(-.17 if pid==E else .17);axs[1].barh(y,vals,color=col,height=.3,label=label)
     profile_values.extend(dict(policy_id=pid,step=int(t),design_mean=float(v),reference_mean=float(b0),relative_change_percent=float(delta)) for t,v,b0,delta in zip(range(1,501),a,base,100*(a/base-1)))
     activity_values.extend(dict(policy_id=pid,window_metric=f,design_mean=float(means[(1,pid)][f]),reference_mean=float(means[(1,B)][f]),relative_change_percent=float(v)) for f,v in zip(fields,vals))
     for v,yy in zip(vals,y):axs[1].text(v+(.04 if v>=0 else -.04),yy,f'{v:+.2f}' if abs(v)>=.01 else f'{v:+.4f}',ha='left' if v>=0 else 'right',va='center',fontsize=13)
    axs[0].set(title='a  Water during the early period',xlabel='Simulation step',ylabel='Mean stock change\n(% of original stock)');axs[0].legend(frameon=False,fontsize=13)
    axs[1].set(title='b  Activity gains across\nearly and late windows',xlabel='Support change\n(% of original support)',yticks=range(4),yticklabels=['Human: first 100','AI: first 100','Human: late period','AI: late period']);axs[1].invert_yaxis();axs[1].set_xlim(-2.8,2.8)
    for ax in axs:ax.axvline(0,color='#aaa',lw=.5) if ax==axs[1] else ax.axhline(0,color='#aaa',lw=.5)
    save(fig,'activity_windows')
    pd.DataFrame(profile_values).to_csv(destination/'formation_stock_values.csv',index=False)
    pd.DataFrame(activity_values).to_csv(destination/'formation_activity_values.csv',index=False)
    # robustness: physical scenario effects compared with payoff effects within each scenario.
    r=pd.read_csv(bundle/'analysis/robustness_comparisons.csv').drop_duplicates(['variant','policy_id']);vs=sorted(r.variant.unique());names={'clustered_ai':'AI concentration','demand_ratio_40':'AI demand ratio 40','demand_ratio_5':'AI demand ratio 5','epsilon_0.05':'Reversal probability 0.05','epsilon_0.2':'Reversal probability 0.2','human_priority_allocation':'Human priority','initial_0.1':'Initial cooperation 0.1','initial_0.9':'Initial cooperation 0.9','larger_grid':'Larger world','long_horizon':'10,000 steps','manuscript_feedback':'Conference renewal law','sequential_planning':'Sequential planning','theta_1.4':'Renewal response 1.4','theta_5':'Renewal response 5','unconditioned_placement':'Unconditioned placement'}
    fig,axs=plt.subplots(1,3,figsize=(11,6.8),layout='constrained');robustbase=r[(r.variant=='initial_0.1')&(r.policy_id==B)].iloc[0]
    for j,v in enumerate(vs):
     q=r[r.variant==v].set_index('policy_id');b0=q.loc[B]
     for k,(field,col,off) in enumerate([('human',blue,-.13),('ai',gray,.13)]):
      base=robustbase[field];x=100*(b0[field]/base-1);axs[0].plot(x,j+off,marker='o' if k==0 else 's',color=col,ms=4)
      d=100*(q.loc[q.index!=B,field]/b0[field]-1);ax=axs[k+1];ax.plot([d.min(),d.max()],[j,j],color=col,lw=2);ax.plot([d.min(),d.max()],[j,j],color=col,marker='|',ls='',ms=7)
    axs[0].set(title='a  Changed assumptions',xlabel='Original-game support\nvs matched default (%)',yticks=range(15),yticklabels=[names[v] for v in vs]);fig.legend(handles=[Line2D([],[],color=blue,marker='o',ls='',label='Human'),Line2D([],[],color=gray,marker='s',ls='',label='AI')],frameon=False,loc='outside lower center',ncol=2)
    for ax,title in zip(axs[1:],['b  Frozen designs: human','c  Frozen designs: AI']):ax.set(title=title,xlabel='Mean-change range\n(% of own original)',yticks=range(15),yticklabels=[])
    for ax in axs:ax.invert_yaxis();ax.axvline(0,color='black',lw=.6);ax.grid(axis='y',alpha=.1)
    save(fig,'assumption_sensitivity')
    r.to_csv(destination/'robustness_values.csv',index=False)
    # all other search settings, with all objectives represented by frontier membership.
    settings=f[['pressure','budget','norm']].drop_duplicates();secondary=settings[~((settings.pressure==1)&(settings.budget==.5)&(settings.norm=='L2'))]
    fig,axs=plt.subplots(3,2,figsize=(9,10.5),layout='constrained')
    for ax,(_,s) in zip(axs.flat,secondary.iterrows()):
     q=f[(f.pressure==s.pressure)&(f.budget==s.budget)&(f.norm==s['norm'])];dots(ax,q,0,1);ax.set(title=f"Pressure {s.pressure:g}; budget {s.budget:g}; {s['norm']} ({int(q.unconstrained_combined_sample_frontier.sum())}/{len(q)})",xlabel='Human support change (%)',ylabel='AI support change (%)')
    save(fig,'other_tradeoffs')
    f.to_csv(destination/'combined_tested_frontiers.csv',index=False)

    coverage=pd.read_csv(D/'coverage_accounting_means.csv');coverage=coverage[(coverage.split=='test')&(coverage.policy_id==B)].sort_values('pressure')
    fig,axs=plt.subplots(1,2,figsize=(9.5,4.5),layout='constrained');x=np.arange(len(coverage))
    axs[0].plot(x,100*coverage.human_mean_cooperation,'o-',color=blue,label='Human');axs[0].plot(x,100*coverage.ai_mean_cooperation,'s--',color=gray,label='AI')
    axs[0].set(title='a  Who makes lower requests?',ylabel='Lower requests (%)');axs[0].legend(frameon=False,fontsize=13)
    axs[1].bar(x,100*coverage.lower_request_contribution,color=blue,label='Lower share of target requested')
    axs[1].bar(x,100*coverage.unfilled_request_contribution,bottom=100*coverage.lower_request_contribution,color=gray,label='Requested but not delivered')
    axs[1].set(title='b  Two parts of the coverage gap',ylabel='Human minus AI coverage\n(percentage points)')
    for ax in axs:ax.set(xticks=x,xticklabels=[f'{p:g}' for p in coverage.pressure],xlabel='Demand relative to renewal');ax.grid(axis='y',alpha=.1)
    handles,legend_labels=axs[1].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='outside lower center',ncol=1,frameon=False,fontsize=12);save(fig,'coverage_components');coverage.to_csv(destination/'coverage_accounting_values.csv',index=False)
    events=pd.read_csv(D/'paired_source_events_means.csv');event_specs=[(1,A,'A'),(1,O,'B'),(1,E,'C'),(1,N,'D'),(.25,'e9883b257b7c3a524aafbff6','Low-pressure design')]
    fig,ax=plt.subplots(figsize=(8.5,4.5),layout='constrained');event_rows=[]
    for j,(p,pid,label) in enumerate(event_specs):
        q=events[(events.pressure==p)&(events.policy_id==pid)].iloc[0];event_rows.append(q)
        ax.barh(j-.16,100*q.relieved_source_fraction,height=.28,color=blue,label='Previously affected sources relieved' if j==0 else None)
        ax.barh(j+.16,-100*q.newly_affected_source_fraction,height=.28,color=gray,label='Previously unaffected sources newly affected' if j==0 else None)
    ax.set(yticks=range(len(event_specs)),yticklabels=[v[2] for v in event_specs],xlabel='Source locations (%): new burdens left, relief right');ax.axvline(0,color='black',lw=.6);ax.invert_yaxis();handles,legend_labels=ax.get_legend_handles_labels();fig.legend(handles,legend_labels,loc='outside lower center',ncol=1,frameon=False,fontsize=12);ax.grid(axis='x',alpha=.1)
    save(fig,'source_events');pd.DataFrame(event_rows).to_csv(destination/'source_event_values.csv',index=False)

    inputs = [bundle/'analysis'/n for n in ['pareto_candidates.csv','paired_outcome_comparisons.csv','robustness_comparisons.csv']]
    inputs += [D/n for n in ['trajectory_condition_means.csv','all_one_action_responses_and_bands.csv','complete_profiles.json','provenance_and_verification.json','paired_resource_budget_means.csv','coverage_accounting_means.csv','paired_source_events_means.csv']]
    record = {'run_id': manifest['run_id'], 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
              'scope': 'Saved-data analysis only. Ratios are ratios of ensemble means; original intervals retain difference units.',
              'primary_tested_designs': len(primary), 'primary_unconstrained_frontier': int(primary.unconstrained_combined_sample_frontier.sum()),
              'figures': sorted(p.name for p in F.glob('*.pdf'))}
    write_json(destination/'plot_manifest.json', record)
    return record
