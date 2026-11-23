"""Tables for the experiments of the paper and an audit of the saved staged search.

Nothing here simulates. Paired statistics use the world as the unit; every
condition of one world shares its landscape and indexed random inputs.
"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import t as student
from .analysis import write_csv
from .config import scenario as configured_scenario
from .paper_experiments import CLASSES, SERIES, grid_name, load_classes, nondominated, search_table
from .metrics import paired_simultaneous_intervals
from .provenance import read_jsonl, sha256, source_hashes, write_json
from .world import World

LOAD_BINS = (0, .4, 2/3, .8, .9, 1., 1.1, 1.25, 1.5, 2., 3., np.inf)
OUTCOMES = ('human_activity', 'ai_activity', 'human_delivered', 'ai_delivered', 'cooperation', 'stock', 'contested_stock',
            'renewal', 'bonus_renewal', 'spilled_renewal', 'below_half_share', 'empty_share', 'contested_empty_share',
            'human_scarcity', 'ai_scarcity', 'human_requested', 'ai_requested', 'human_cooperation', 'ai_cooperation',
            'human_request_met', 'ai_request_met')


def summary(values):
    """Mean, standard error and a descriptive t interval over independent worlds."""
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(x):
        return dict(worlds=0, mean=None, standard_error=None, lower=None, upper=None, positive=None)
    error = float(x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else None
    half = float(student.ppf(.975, len(x)-1)*error) if error is not None else None
    return dict(worlds=len(x), mean=float(x.mean()), standard_error=error,
                lower=None if half is None else float(x.mean()-half), upper=None if half is None else float(x.mean()+half),
                positive=int((x > 0).sum()))


def longest_run(mask):
    best, current = np.zeros(mask.shape[1], dtype=int), np.zeros(mask.shape[1], dtype=int)
    for row in mask:
        current = np.where(row, current+1, 0)
        best = np.maximum(best, current)
    return best


# audit of the saved staged search ---------------------------------------------------------

def audit_search(bundle, output, worlds=320):
    """What the executed search scored, and whether those scores could separate designs."""
    bundle, output = Path(bundle), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((bundle/'resolved_config.json').read_text())
    manifest = json.loads((bundle/'manifest.json').read_text())
    candidates = read_jsonl(bundle/'training_candidates.jsonl')
    tests = json.loads((bundle/'test_comparisons.json').read_text())
    stress, burn = config['objective']['persistent_local_stress'], config['objective']['excluded_initial_steps']
    ranges = []
    for pressure in sorted({r['pressure'] for r in candidates}):
        rows = [r for r in candidates if r['pressure'] == pressure and r['policy']['observation'] == 'local' and not r['policy']['forced']]
        original = next(r for r in rows if r['policy']['u'] == 0 and r['policy']['v'] == 0)
        row = dict(load=pressure, payoff_designs=len(rows), search_worlds_per_design=original['count'],
                   designs_with_event_probability_below_one=sum(r['stress_probability'] < 1 for r in rows),
                   lowest_event_probability=min(r['stress_probability'] for r in rows))
        for key, field, per_world in [('human_activity', 'mean_human_service', 'human_services'), ('ai_activity', 'mean_ai_service', 'ai_services'),
                                      ('depletion_score', 'mean_loss', 'losses')]:
            values = np.array([r[field] for r in rows])
            paired = np.array([np.mean(np.array(r[per_world])-np.array(original[per_world])) for r in rows])
            row.update({f'{key}_original': original[field], f'{key}_lowest': float(values.min()), f'{key}_highest': float(values.max()),
                        f'{key}_largest_gain_over_original': float(paired.max() if key != 'depletion_score' else -paired.min()),
                        f'{key}_between_world_sd_original': float(np.std(original[per_world], ddof=1))})
        ranges.append(row)
    write_csv(output/'search_score_ranges.csv', ranges)
    forced = []
    for entry in json.loads((bundle/'reference_characterization.json').read_text()):
        for name, r in entry['policies'].items():
            forced.append(dict(load=entry['pressure'], rule=name, worlds=r['count'], human_activity=r['mean_human_service'],
                               ai_activity=r['mean_ai_service'], depletion_score=r['mean_loss'], event_probability=r['stress_probability']))
    write_csv(output/'forced_action_scores.csv', forced)
    references = {}
    for row in tests['problems']:
        references.setdefault(row['problem']['pressure'], row['families']['reference'][0])
    account, geometry, sources = [], [], []
    for pressure, reference in sorted(references.items()):
        physical = configured_scenario(config, pressure)
        records, per_world = [], []
        for evaluation, world_id, event in list(zip(reference['realization_ids'], reference['world_ids'], reference['stress_events']))[:worlds]:
            world = World.load(bundle/'worlds'/f'{world_id}.npz')
            with np.load(bundle/'trajectories'/f'{evaluation}.npz', allow_pickle=False) as saved:
                stocks, renewal, requests = saved['stocks'], saved['resource_renewal'], saved['agent_reference_request']
            load, classes = load_classes(world, requests, physical)
            acc = world.access.any(axis=0)
            n = stocks/world.capacity
            run = longest_run(n[burn:] < stress['reference'])
            late = n[-500:].mean(axis=0)
            first, last = n[burn:burn+500].mean(axis=0), late
            held = run >= stress['duration_steps']
            if bool(held[acc].any()) != bool(event):
                raise ArithmeticError('recomputed stock event differs from the saved result')
            ratio = renewal[burn:].mean(axis=0)/world.renewal
            for j in np.flatnonzero(acc):
                records.append((load[j], classes[j], late[j], held[j], ratio[j], first[j], last[j], world.renewal[j]))
            per_world.append(dict(accessible=int(acc.sum()), overloaded=int((acc & (classes == 2)).sum()),
                                  held=int(held[acc].sum()), held_overloaded=int((held & acc & (classes == 2)).sum()),
                                  held_contested=int((held & acc & (classes == 1)).sum()), event=bool(event),
                                  baseline=float(world.renewal[acc].sum()), actual=float(renewal[burn:, acc].sum(axis=1).mean()),
                                  bonus=float((np.maximum(ratio-1, 0)*world.renewal)[acc].sum())))
        r = np.array(records)
        for low, high in zip(LOAD_BINS[:-1], LOAD_BINS[1:]):
            member = (r[:, 0] >= low) & (r[:, 0] < high)
            if member.any():
                account.append(dict(load=pressure, local_load_from=low, local_load_below=None if np.isinf(high) else high,
                                    share_of_accessible_sources=float(member.mean()), mean_late_stock=float(r[member, 2].mean()),
                                    full_share=float((r[member, 2] > .9).mean()), empty_share=float((r[member, 2] < .1).mean()),
                                    intermediate_share=float(((r[member, 2] >= .1) & (r[member, 2] <= .9)).mean()),
                                    held_below_reference_share=float(r[member, 3].mean()),
                                    actual_to_baseline_renewal=float(np.average(r[member, 4], weights=r[member, 7])),
                                    class_changed_between_halves=float((np.abs(r[member, 5]-r[member, 6]) > .2).mean())))
        w = {key: np.array([p[key] for p in per_world], dtype=float) for key in per_world[0]}
        geometry.append(dict(load=pressure, worlds=len(per_world), accessible_sources=float(w['accessible'].mean()),
                             overloaded_sources=float(w['overloaded'].mean()),
                             worlds_with_an_overloaded_source=float((w['overloaded'] > 0).mean()),
                             sources_held_below_reference=float(w['held'].mean()), fewest_held=int(w['held'].min()),
                             held_and_overloaded=float(w['held_overloaded'].mean()), held_and_contested=float(w['held_contested'].mean()),
                             observed_event_probability=float(w['event'].mean()),
                             actual_to_baseline_renewal=float((w['actual']/w['baseline']).mean()),
                             # source-level excess of actual over baseline renewal; full sources hide part of it.
                             renewal_above_baseline_share_of_actual=float((w['bonus']/w['actual']).mean())))
        sources.append((pressure, r))
    write_csv(output/'source_states_by_local_load.csv', account)
    write_csv(output/'stock_event_geometry.csv', geometry)
    _plot_source_states(sources, output)
    result = dict(bundle_run_id=manifest['run_id'], configuration_id=manifest['configuration_id'], worlds_per_load=worlds,
                  stock_event=dict(reference=stress['reference'], duration_steps=stress['duration_steps'],
                                   permitted_probability=stress['maximum_probability']),
                  score_ranges=ranges, stock_event_geometry=geometry,
                  inputs={name: sha256(bundle/name) for name in ['training_candidates.jsonl', 'test_comparisons.json',
                                                                'reference_characterization.json', 'resolved_config.json']},
                  analysis_source=source_hashes()['src/human_ai_commons/paper_tables.py'],
                  interpretation='Saved-data description of the executed search. Overloaded sources are defined from geometry and '
                                 'requests alone; their count bounds the stock event from below for every payoff design because '
                                 'source selection does not depend on payoffs.')
    write_json(output/'search_audit.json', result)
    return dict(output=str(output), files=sorted(p.name for p in output.iterdir()), stock_event_geometry=geometry)


def _style():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12,
                         'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    return plt


def _save(plt, fig, folder, name):
    fig.savefig(folder/(name+'.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    fig.savefig(folder/(name+'.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_source_states(sources, output):
    plt = _style()
    fig, axes = plt.subplots(1, len(sources), figsize=(4.2*len(sources), 3.8), sharey=True, layout='constrained', squeeze=False)
    generator = np.random.default_rng(0)
    for ax, (pressure, r) in zip(axes[0], sources):
        keep = generator.choice(len(r), size=min(len(r), 6000), replace=False)
        ax.scatter(np.minimum(r[keep, 0], 3), r[keep, 2], s=4, alpha=.25, color='#33658a', linewidths=0)
        for x in [2/3, 1.25]:
            ax.axvline(x, color='0.35', linewidth=.8, linestyle='--')
        ax.set_title(f'total ordinary demand / renewal = {pressure:g}')
    fig.supxlabel('expected ordinary load on the source (requests / own baseline renewal; values of 3 or more shown at 3)', fontsize=12)
    axes[0][0].set_ylabel('mean stock, last 500 steps\n(fraction of capacity)')
    _save(plt, fig, output, 'source_stock_by_local_load')


# experiments of the paper ----------------------------------------------------------------------

EXPOSURES = ('only_overloaded', 'mixed', 'no_overloaded')


def _paired(rows, reference, outcomes=OUTCOMES):
    """Differences from the reference row of the same world, per outcome."""
    result = {}
    for outcome in outcomes:
        values = [row[outcome]-reference[row['world_index']][outcome] for row in rows
                  if row['world_index'] in reference and row.get(outcome) is not None and reference[row['world_index']].get(outcome) is not None]
        result[outcome] = summary(values)
    return result


def _flat(prefix, stats):
    return {f'{prefix}{name}': stats[name] for name in ['mean', 'standard_error', 'lower', 'upper', 'positive']}


def array_file(output, experiment, cell, index, series=False):
    suffix = '-series' if series else ''
    return output/'arrays'/f'{experiment}-{cell}-{index:04d}{suffix}.npz'


def report(output):
    output = Path(output)
    rows = read_jsonl(output/'rows.jsonl')
    for row in rows:
        # the share of what a group asked for that it received; demand met instead counts every agent at its higher request.
        for kind in ['human', 'ai']:
            if row.get(f'{kind}_requested') and row.get(f'{kind}_delivered') is not None:
                row[f'{kind}_request_met'] = row[f'{kind}_delivered']/row[f'{kind}_requested']
    manifest = json.loads((output/'manifest.json').read_text())
    config = manifest['configuration']
    folder = output/'report'
    folder.mkdir(exist_ok=True)
    tables = {}
    continuity = [r for r in rows if r['experiment'] == 'continuity']
    if continuity:
        tables.update(_continuity(continuity, config))
    for experiment, function in [('interventions', _interventions), ('feedback', _feedback), ('search', _search), ('search_test', _search_test)]:
        selected = [r for r in rows if r['experiment'] == experiment]
        for cell in dict.fromkeys(r['cell'] for r in selected):
            extra = (config,) if experiment.startswith('search') else ()
            made = function([r for r in selected if r['cell'] == cell], output, cell, *extra)
            for name, table in made.items():
                tables.setdefault(name, []).extend(dict(cell=cell) | row for row in table)
    placement = _placement_effects(rows, config)
    if placement:
        tables['intervention_placement_effects.csv'] = placement
    if config.get('confirmatory', {}).get('contrasts'):
        tables['confirmatory_contrasts.csv'] = _confirmatory(rows, config['confirmatory'])
    for name, table in tables.items():
        write_csv(folder/name, table)
    write_json(folder/'report_manifest.json', dict(run_id=manifest['run_id'], protocol=manifest['protocol'], role=manifest['role'],
               rows_sha256=sha256(output/'rows.jsonl'), simulation_source=manifest['source_sha256'],
               analysis_source=source_hashes()['src/human_ai_commons/paper_tables.py'], tables=sorted(tables)))
    return dict(report=str(folder), tables=sorted(tables), worlds=config['worlds'], role=manifest['role'])


def _confirmatory(rows, spec):
    """Pre-declared paired contrasts with one error allowance shared across all of them.

    The primary interval is a paired t interval at confidence 1 - alpha/K for K
    declared contrasts. A distribution-free bounded-mean interval with the same
    simultaneous coverage is reported beside it for outcomes that lie in [0,1].
    A world in which the outcome is undefined on either side, such as the stock of
    contested sources in a world that has none, is left out of that contrast; the
    table gives the number of worlds each contrast used.
    """
    alpha, margin = spec.get('alpha', .05), spec.get('smallest_difference_of_interest', .01)
    table, columns, used = [], [], []
    for contrast in spec['contrasts']:
        pool = [r for r in rows if r['experiment'] == contrast['experiment'] and r['cell'] == contrast.get('cell', 'reference')]
        # a contrast between two settings of the same worlds names the setting of its reference side.
        other = [r for r in rows if r['experiment'] == contrast['experiment'] and r['cell'] == contrast.get('reference_cell', contrast.get('cell', 'reference'))]
        label = contrast['name']

        def pick(selection, pool=pool):
            found = {}
            for r in pool:
                if all(r.get(key) == value for key, value in selection.items()):
                    if r['world_index'] in found:
                        raise ValueError(f'contrast {label} does not select one row per world')
                    found[r['world_index']] = r
            return found

        first, second = pick(contrast['select']), pick(contrast['reference'], other)
        outcome = contrast['outcome']
        worlds = sorted(w for w in set(first) & set(second) if first[w][outcome] is not None and second[w][outcome] is not None)
        if not worlds:
            raise ValueError(f'contrast {label} matches no saved rows')
        values = np.array([first[w][outcome]-second[w][outcome] for w in worlds], dtype=float)
        columns.append(values)
        used.append(worlds)
        table.append(dict(name=contrast['name'], question=contrast.get('question', ''), experiment=contrast['experiment'],
                          cell=contrast.get('cell', 'reference'), reference_cell=contrast.get('reference_cell', contrast.get('cell', 'reference')),
                          outcome=contrast['outcome'], worlds=len(worlds), first_worlds=len(first), second_worlds=len(second),
                          first_mean=float(np.mean([first[w][outcome] for w in worlds])),
                          second_mean=float(np.mean([second[w][outcome] for w in worlds])),
                          difference=float(values.mean()), standard_error=float(values.std(ddof=1)/np.sqrt(len(values))),
                          worlds_positive=int((values > 0).sum())))
    count = len(table)
    # the distribution-free interval needs every contrast on the same worlds, so it uses those that all of them share.
    bounded, shared = None, sorted(set.intersection(*map(set, used))) if used else []
    if len(shared) > 1 and all(np.all(np.abs(c) <= 1) for c in columns):
        common = [values[np.isin(worlds, shared)] for values, worlds in zip(columns, used)]
        bounded = paired_simultaneous_intervals(np.column_stack(common), confidence=1-alpha)
    for position, (row, values) in enumerate(zip(table, columns)):
        if len(values) < 2:
            raise ValueError('confirmatory contrasts need at least two worlds')
        half = float(student.ppf(1-alpha/(2*count), len(values)-1)*row['standard_error'])
        low, high = row['difference']-half, row['difference']+half
        row.update(lower=low, upper=high, simultaneous_confidence=1-alpha, declared_contrasts=count,
                   smallest_difference_of_interest=margin,
                   reading=('differs from zero' if low > 0 or high < 0 else 'no difference resolved')
                           + ('; smaller than the margin' if -margin < low and high < margin else
                              '; larger than the margin' if low >= margin or high <= -margin else ''),
                   bounded_lower=None if bounded is None else bounded[position]['lower'],
                   bounded_upper=None if bounded is None else bounded[position]['upper'])
    return table


def _continuity(rows, config):
    usual = config['world']['capacity']
    conditions = list(dict.fromkeys(r['condition'] for r in rows))
    windows = list(dict.fromkeys(r['window'] for r in rows))
    table = []
    for condition in conditions:
        for window in windows:
            group = [r for r in rows if r['condition'] == condition and r['window'] == window]
            first = group[0]
            row = dict(condition=condition, group=first['group'], changes=first['changes'], requests=first['requests'],
                       placement=first['placement'], allocation=first['allocation'], renewal_law=first['feedback'],
                       capacity=first['capacity'], density=first['density'], initial_stock=first['initial_stock'],
                       window=window, window_start=first['window_start'], window_stop=first['window_stop'], worlds=len(group))
            for key in ['relative_load', 'total_ordinary_request', 'accessible_baseline_renewal', 'initial_stored_volume',
                        'agents_without_access', 'delivery_ceiling_baseline_renewal', 'delivery_ceiling_largest_renewal',
                        'cooperation', 'stock', 'stock_all_sources', 'human_activity', 'ai_activity', 'human_delivered',
                        'ai_delivered', 'renewal', 'spilled_renewal', 'bonus_renewal', 'empty_share', 'intermediate_share', 'full_share',
                        'underused_sources', 'contested_sources', 'overloaded_sources']:
                stats = summary([r.get(key) for r in group])
                row[key], row[key+'_se'] = stats['mean'], stats['standard_error']
            table.append(row)
    effects = []
    for label in ['fixed_requests', 'journal']:
        for window in windows:
            reference = {r['world_index']: r for r in rows if r['condition'] == f'storage_{label}_K{usual}_full' and r['window'] == window}
            for condition in [c for c in conditions if c.startswith(f'storage_{label}_')]:
                group = [r for r in rows if r['condition'] == condition and r['window'] == window]
                paired = _paired(group, reference, ('cooperation', 'stock', 'human_delivered', 'ai_delivered', 'human_activity', 'ai_activity', 'empty_share'))
                row = dict(requests=label, condition=condition, capacity=group[0]['capacity'], initial_stock=group[0]['initial_stock'],
                           window=window, compared_with=f'storage_{label}_K{usual}_full', worlds=len(group))
                for outcome, stats in paired.items():
                    row.update(_flat(outcome+'_difference_', stats))
                effects.append(row)
    return {'continuity_conditions.csv': table, 'continuity_storage_effects.csv': effects}


def load_series(output, experiment, cell, index):
    with np.load(array_file(output, experiment, cell, index, series=True), allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files if key != 'columns'}


def mean_series(output, experiment, cell, worlds, key, column):
    stack = [load_series(output, experiment, cell, index)[key][:, SERIES.index(column)] for index in worlds]
    return np.mean(stack, axis=0)


def _measure_label(row):
    """A reader's name for one condition, from what it changes."""
    parts = []
    if row.get('selection') == 'fullest':
        parts.append('price, fullest source')
    elif row.get('selection') == 'priced':
        parts.append('price, strength {:g}'.format(row['price']))
    if row.get('floor'):
        parts.append('cap at {:g}%'.format(100*row['floor']))
    scope = ', all agents' if parts and row.get('regulated') != 'ai' else ''
    if row.get('allocation_override') == 'human_priority':
        parts.append('human agents served first')
    if parts:
        moved = '' if row['r_c'] == row['r_d'] == 0 else ', switching level %.2f' % row['q_c']
        return ' and '.join(parts)+scope+moved
    if row.get('admissible') and row['r_c'] == row['r_d']:
        return 'switching level %.2f' % row['q_c']
    return row['design'].replace('_', ' ')


def _resolved(config, cell):
    """Everything that defines a setting except where the AI agents are placed."""
    spec = config.get('cells', {}).get(cell, {})
    return (config['scenario'] | spec.get('scenario', {}), config['world'] | spec.get('world_spec', {}), spec.get('requests', 'scaled'))


def _placement_effects(rows, config):
    """The original game under sited placement against random placement of the same worlds, per window."""
    cells = config.get('cells', {})
    kept = [r for r in rows if r['experiment'] == 'interventions' and r['condition'] == 'unchanged' and r['timing'] == 'after_development']
    present = list(dict.fromkeys(r['cell'] for r in kept))
    table = []
    for cell in present:
        spec = cells.get(cell, {})
        if spec.get('siting', 'random') == 'random':
            continue
        partner = next((name for name in present if cells.get(name, {}).get('siting', 'random') == 'random'
                        and _resolved(config, name) == _resolved(config, cell)), None)
        if partner is None:
            continue
        for window in dict.fromkeys(r['window'] for r in kept if r['cell'] == cell):
            sited = [r for r in kept if r['cell'] == cell and r['window'] == window]
            reference = {r['world_index']: r for r in kept if r['cell'] == partner and r['window'] == window}
            row = dict(cell=cell, random_placement_cell=partner, sited_share=spec.get('sited_share', 1.), window=window,
                       worlds=len({r['world_index'] for r in sited} & set(reference)))
            for outcome, stats in _paired(sited, reference).items():
                row.update(_flat(outcome+'_difference_', stats))
                row[outcome+'_random_placement'] = summary([reference[r['world_index']].get(outcome) for r in sited if r['world_index'] in reference])['mean']
            # what the placement itself changes, before any behavior: reach, accessible supply and the sustained-delivery ceiling.
            for key in ['relative_load', 'accessible_sources', 'accessible_baseline_renewal', 'delivery_ceiling_baseline_renewal',
                        'delivery_ceiling_largest_renewal', 'human_reachable_sources', 'ai_reachable_sources']:
                row[key+'_sited'] = summary([r.get(key) for r in sited])['mean']
                row[key+'_under_random_placement'] = summary([r.get(key) for r in reference.values()])['mean']
            table.append(row)
    return table


def _interventions(rows, output, cell):
    designs = [d for d in dict.fromkeys(r['condition'] for r in rows) if d != 'unchanged']
    table, distribution = [], []
    shares = tuple(f'{kind}_{level}_activity' for kind in ['human', 'ai'] for level in EXPOSURES)
    for timing, reference_timing in [('after_development', 'after_development'), ('from_initialization', 'from_initialization'),
                                     ('from_initialization_same_time', 'after_development')]:
        source_timing = 'from_initialization' if timing.startswith('from') else timing
        windows = list(dict.fromkeys(r['window'] for r in rows if r['condition'] != 'unchanged' and r['timing'] == source_timing
                                     and (r['window'].startswith('same_time:') == (timing == 'from_initialization_same_time'))))
        for window in windows:
            plain = window.replace('same_time:', '')
            reference = {r['world_index']: r for r in rows if r['condition'] == 'unchanged' and r['timing'] == reference_timing and r['window'] == plain}
            for design in designs:
                group = [r for r in rows if r['condition'] == design and r['timing'] == source_timing and r['window'] == window]
                if not group:
                    continue
                first = group[0]
                head = dict(design=design, timing=timing, window=plain, window_start=first['window_start'], window_stop=first['window_stop'],
                            r_c=first['r_c'], r_d=first['r_d'], q_c=first['q_c'], q_d=first['q_d'], forced=first['forced'],
                            cost_l2=first['cost_l2'], consequence_weight=first.get('consequence_weight', 0.),
                            reversal_override=first.get('reversal_override'), admissible=first.get('admissible'),
                            theta=first.get('theta'), relative_load=summary([r['relative_load'] for r in group])['mean'],
                            requests=first.get('requests'), worlds=len(group), measure=_measure_label(first),
                            **{key: first.get(key) for key in ['selection', 'price', 'floor', 'regulated', 'allocation_override']})
                row = dict(head)
                for outcome, stats in _paired(group, reference).items():
                    row.update(_flat(outcome+'_difference_', stats))
                    row[outcome+'_unchanged'] = summary([reference[r['world_index']].get(outcome) for r in group])['mean']
                table.append(row)
                if timing == 'after_development':
                    for outcome, stats in _paired(group, reference, shares+('human_activity_p10', 'ai_activity_p10')).items():
                        kind = outcome.split('_')[0]
                        distribution.append(dict(design=design, window=plain, agent_type=kind,
                                                 group='lowest_tenth_of_agents' if outcome.endswith('p10') else outcome[len(kind)+1:].replace('_activity', ''),
                                                 agents=summary([r.get(outcome.replace('_activity', '_agents')) for r in group])['mean'],
                                                 activity_unchanged=summary([reference[r['world_index']].get(outcome) for r in group])['mean'],
                                                 **_flat('difference_', stats)))
    states = []
    worlds = sorted({r['world_index'] for r in rows})
    for index in worlds:
        with np.load(array_file(output, 'interventions', cell, index), allow_pickle=False) as saved:
            classes, before = saved['unchanged::classes'], saved['unchanged::developed_stock']
            base = saved['unchanged::final_stock']
            for design in designs:
                final = saved[f'{design}::final_stock']
                for code, label in enumerate(CLASSES):
                    member = classes == code
                    if member.any():
                        states.append(dict(world_index=index, design=design, source_class=label, sources=int(member.sum()),
                                           empty_before=int((before[member] < .1).sum()), empty_unchanged=int((base[member] < .1).sum()),
                                           empty_design=int((final[member] < .1).sum()),
                                           emptied_by_design=int(((base[member] >= .1) & (final[member] < .1)).sum()),
                                           restored_by_design=int(((base[member] < .1) & (final[member] >= .1)).sum()),
                                           stock_unchanged=float(base[member].mean()), stock_design=float(final[member].mean())))
    pooled = []
    for design in designs:
        for label in CLASSES:
            group = [s for s in states if s['design'] == design and s['source_class'] == label]
            if group:
                pooled.append(dict(design=design, source_class=label, worlds=len(group),
                                   **{key: float(np.mean([g[key] for g in group])) for key in
                                      ['sources', 'empty_before', 'empty_unchanged', 'empty_design', 'emptied_by_design', 'restored_by_design',
                                       'stock_unchanged', 'stock_design']}))
    return {'intervention_effects.csv': table, 'intervention_source_states.csv': pooled, 'intervention_distribution.csv': distribution}


def _feedback(rows, output, cell):
    worlds = sorted({r['world_index'] for r in rows})
    made = {}
    hysteresis = [r for r in rows if r.get('part') == 'hysteresis']
    if hysteresis:
        windows = list(dict.fromkeys(r['window'] for r in hysteresis))
        cases = list(dict.fromkeys((r['design'], r['renewal_rule']) for r in hysteresis))
        described = hysteresis[0]
        table = []
        for design, renewal in cases:
            per_world = []
            for index in worlds:
                with np.load(array_file(output, 'feedback', cell, index), allow_pickle=False) as saved:
                    classes = saved[f'hysteresis|{design}|{renewal}|full::classes']
                    acc = classes >= 0
                    first = {start: saved[f'hysteresis|{design}|{renewal}|{start}::{windows[0]}'] for start in ['full', 'empty']}
                    last = {start: saved[f'hysteresis|{design}|{renewal}|{start}::{windows[-1]}'] for start in ['full', 'empty']}
                    activity = {start: saved[f'hysteresis|{design}|{renewal}|{start}::agent_activity'] for start in ['full', 'empty']}
                gap = np.abs(last['full']-last['empty'])
                drift = np.abs(last['full']-first['full'])
                contested = max(int((classes == 1).sum()), 1)
                per_world.append(dict(accessible=int(acc.sum()), contested=int((classes == 1).sum()), overloaded=int((classes == 2).sum()),
                                      differ_last=int((gap[acc] > .2).sum()), differ_first=int((np.abs(first['full']-first['empty'])[acc] > .2).sum()),
                                      differ_last_contested=int((gap[classes == 1] > .2).sum()),
                                      differ_last_share_of_contested=float((gap[classes == 1] > .2).sum()/contested),
                                      same_run_drift=int((drift[acc] > .2).sum()),
                                      lower_from_empty=int(((last['full']-last['empty'])[acc] > .2).sum()),
                                      higher_from_empty=int(((last['empty']-last['full'])[acc] > .2).sum()),
                                      mean_stock_gap=float((last['full']-last['empty'])[acc].mean()),
                                      activity_gap=float(activity['full'].mean()-activity['empty'].mean())))
            row = dict(design=design, renewal_rule=renewal, worlds=len(per_world), first_window=windows[0], last_window=windows[-1],
                       theta=described.get('theta'), relative_load=summary([r['relative_load'] for r in hysteresis])['mean'],
                       requests=described.get('requests'))
            for key in per_world[0]:
                stats = summary([p[key] for p in per_world])
                row[key], row[key+'_se'] = stats['mean'], stats['standard_error']
            table.append(row)
        made['feedback_start_dependence.csv'] = table
        effects = []
        for design, renewal in cases:
            for window in windows:
                reference = {r['world_index']: r for r in hysteresis if (r['design'], r['renewal_rule'], r['initial_stock'], r['window']) == (design, renewal, 'full', window)}
                group = [r for r in hysteresis if (r['design'], r['renewal_rule'], r['initial_stock'], r['window']) == (design, renewal, 'empty', window)]
                row = dict(design=design, renewal_rule=renewal, window=window, contrast='empty start minus full start', worlds=len(group))
                for outcome, stats in _paired(group, reference).items():
                    row.update(_flat(outcome+'_difference_', stats))
                effects.append(row)
        made['feedback_start_effects.csv'] = effects
        # the sources that depend on the start under the original game, followed under every design and renewal rule.
        dependent = []
        if ('unchanged', 'coupled') in cases:
            name = 'hysteresis|unchanged|coupled'
            for index in worlds:
                with np.load(array_file(output, 'feedback', cell, index), allow_pickle=False) as saved:
                    classes, load = saved[f'{name}|full::classes'], saved[f'{name}|full::load']
                    gap = np.abs(saved[f'{name}|full::{windows[-1]}']-saved[f'{name}|empty::{windows[-1]}'])
                    for design, renewal in cases:
                        late = {start: saved[f'hysteresis|{design}|{renewal}|{start}::{windows[-1]}'] for start in ['full', 'empty']}
                        dependent.extend(dict(world_index=index, source=int(j), expected_load=float(load[j]), source_class=CLASSES[int(classes[j])],
                                              design=design, renewal_rule=renewal, stock_from_full=float(late['full'][j]),
                                              stock_from_empty=float(late['empty'][j])) for j in np.flatnonzero((gap > .2) & (classes >= 0)))
        if dependent:
            made['feedback_start_dependent_sources.csv'] = dependent
    control = [r for r in rows if r.get('part') == 'control']
    if control:
        table = []
        for window in list(dict.fromkeys(r['window'] for r in control)):
            for renewal in list(dict.fromkeys(r['renewal_rule'] for r in control)):
                reference = {r['world_index']: r for r in control if (r['design'], r['renewal_rule'], r['window']) == ('unchanged', renewal, window)}
                coupled = {r['world_index']: r for r in control if (r['design'], r['renewal_rule'], r['window']) == ('unchanged', 'coupled', window)}
                for design in list(dict.fromkeys(r['design'] for r in control)):
                    group = [r for r in control if (r['design'], r['renewal_rule'], r['window']) == (design, renewal, window)]
                    against, contrast = (coupled, 'renewal control minus coupled renewal, unchanged payoffs') if design == 'unchanged' else (
                        reference, 'design minus unchanged payoffs under the same renewal rule')
                    if design == 'unchanged' and renewal == 'coupled':
                        continue
                    row = dict(design=design, renewal_rule=renewal, window=window, contrast=contrast, worlds=len(group))
                    for outcome, stats in _paired(group, against).items():
                        row.update(_flat(outcome+'_difference_', stats))
                    table.append(row)
        made['feedback_renewal_controls.csv'] = table
    recovery = [r for r in rows if r.get('part') == 'recovery']
    if recovery:
        table, lasting = [], []
        disturbances = [d for d in dict.fromkeys(r['disturbance'] for r in recovery) if d != 'none']
        for window in list(dict.fromkeys(r['window'] for r in recovery)):
            for design in list(dict.fromkeys(r['design'] for r in recovery)):
                reference = {r['world_index']: r for r in recovery if (r['design'], r['disturbance'], r['window']) == (design, 'none', window)}
                for disturbance in disturbances:
                    group = [r for r in recovery if (r['design'], r['disturbance'], r['window']) == (design, disturbance, window)]
                    row = dict(design=design, disturbance=disturbance, window=window, window_start=group[0]['window_start'],
                               window_stop=group[0]['window_stop'], contrast='disturbed minus undisturbed, same payoffs', worlds=len(group))
                    for outcome, stats in _paired(group, reference).items():
                        row.update(_flat(outcome+'_difference_', stats))
                    table.append(row)
        for design in list(dict.fromkeys(r['design'] for r in recovery)):
            for disturbance in disturbances:
                per_world = []
                for index in worlds:
                    with np.load(array_file(output, 'feedback', cell, index), allow_pickle=False) as saved:
                        classes = saved[f'recovery|{design}|none::classes']
                        calm, hit = saved[f'recovery|{design}|none::final_stock'], saved[f'recovery|{design}|{disturbance}::final_stock']
                    acc = classes >= 0
                    per_world.append(dict(still_lower=int(((calm-hit)[acc] > .2).sum()), still_higher=int(((hit-calm)[acc] > .2).sum()),
                                          contested_still_lower=int(((calm-hit)[classes == 1] > .2).sum()),
                                          mean_stock_gap=float((hit-calm)[acc].mean())))
                row = dict(design=design, disturbance=disturbance, worlds=len(per_world), window='last window after the disturbance')
                for key in per_world[0]:
                    stats = summary([p[key] for p in per_world])
                    row[key], row[key+'_se'] = stats['mean'], stats['standard_error']
                lasting.append(row)
        made['feedback_recovery.csv'] = table
        made['feedback_lasting_changes.csv'] = lasting
    return made


SEARCH_OUTCOMES = ('human_activity', 'ai_activity', 'cooperation', 'stock', 'contested_stock', 'contested_empty_share', 'renewal')


def _search(rows, output, cell, config):
    """The payoff grid in one cell: each design's paired difference from the original game."""
    settings = config['search']
    original = grid_name(0., 0.)
    windows = [f'{a}-{z}' for a, z in settings['windows_after']]
    by_design = {row['design']: row for row in search_table(rows, config, cell)}
    fronts = {budget: {row['design'] for row in nondominated(list(by_design.values()), budget)} for budget in settings['budgets']}
    chosen = set()
    if (output/'selected_designs.json').exists():
        chosen = {d['design'] for d in json.loads((output/'selected_designs.json').read_text())[cell]}
    table = []
    for design in dict.fromkeys(r['condition'] for r in rows):
        first = next(r for r in rows if r['condition'] == design)
        row = dict(design=design, r_c=first['r_c'], r_d=first['r_d'], q_c=first['q_c'], q_d=first['q_d'], cost_l2=first['cost_l2'],
                   forced=first['forced'], admissible=first['admissible'], theta=first['theta'],
                   worlds=len({r['world_index'] for r in rows if r['condition'] == design}))
        if design in by_design:
            row.update(meets_requirement=by_design[design]['meets_requirement'], selected_for_test=design in chosen,
                       **{f'nondominated_within_budget_{budget:g}': design in members for budget, members in fronts.items()})
        for window in windows:
            reference = {r['world_index']: r for r in rows if r['condition'] == original and r['window'] == window}
            group = [r for r in rows if r['condition'] == design and r['window'] == window]
            for outcome, stats in _paired(group, reference, SEARCH_OUTCOMES).items():
                row[f'{outcome}_{window}'] = summary([r[outcome] for r in group])['mean']
                row.update(_flat(f'{outcome}_{window}_difference_', stats))
        table.append(row)
    return {'search_map.csv': table}


def summary_mean(values):
    return summary(values)['mean']


def _search_test(rows, output, cell, config):
    """Designs frozen from the search against the original game on worlds the search never used."""
    settings = config['search']
    original = grid_name(0., 0.)
    selected = json.loads((output/'selected_designs.json').read_text())
    items = [*settings['objectives'], settings['requirement']]
    count = max(1, sum(len(designs) for designs in selected.values())*len(items))
    alpha, margin = config['confirmatory']['alpha'], config['confirmatory']['smallest_difference_of_interest']
    table = []
    for design in [d['design'] for d in selected[cell]]:
        first = next(r for r in rows if r['condition'] == design)
        for item in items:
            reference = {r['world_index']: r for r in rows if r['condition'] == original and r['window'] == item['window']}
            group = [r for r in rows if r['condition'] == design and r['window'] == item['window']]
            values = np.array([r[item['outcome']]-reference[r['world_index']][item['outcome']] for r in group], dtype=float)
            error = float(values.std(ddof=1)/np.sqrt(len(values)))
            half = float(student.ppf(1-alpha/(2*count), len(values)-1)*error)
            low, high = float(values.mean())-half, float(values.mean())+half
            table.append(dict(design=design, r_c=first['r_c'], r_d=first['r_d'], q_c=first['q_c'], q_d=first['q_d'], cost_l2=first['cost_l2'],
                              outcome=item['outcome'], window=item['window'], role='requirement' if item is settings['requirement'] else 'objective',
                              worlds=len(values), original_mean=summary_mean([reference[r['world_index']][item['outcome']] for r in group]),
                              difference=float(values.mean()), standard_error=error, lower=low, upper=high, worlds_positive=int((values > 0).sum()),
                              simultaneous_confidence=1-alpha, declared_contrasts=count, smallest_difference_of_interest=margin,
                              reading=('differs from zero' if low > 0 or high < 0 else 'no difference resolved')
                                      + ('; smaller than the margin' if -margin < low and high < margin else
                                         '; larger than the margin' if low >= margin or high <= -margin else '')))
    return {'search_test_effects.csv': table}


