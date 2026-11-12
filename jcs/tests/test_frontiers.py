from dataclasses import asdict
import csv
import json
import numpy as np
from human_ai_commons.frontiers import nondominated, select, spread, grid_search, values
from human_ai_commons.payoffs import Policy
from human_ai_commons.config import load_config, budget_plan
from human_ai_commons.experiments import Evaluation
from human_ai_commons.provenance import ROOT
from human_ai_commons import analysis
from human_ai_commons.frontiers import assess_constraints


def candidate(u, human, ai, depletion, feasible=True):
    policy = Policy(u=u)
    return dict(policy=asdict(policy), policy_id=policy.id, mean_human_service=human,
                mean_ai_service=ai, mean_loss=depletion, search_feasible=feasible,
                stress_probability_violation=-.1 if feasible else .2)


def test_tradeoffs_survive_even_when_a_combined_score_prefers_the_middle():
    rows = [candidate(0, .9, .4, .2), candidate(.1, .65, .65, .2), candidate(.2, .4, .9, .2)]
    dominated = candidate(.3, .3, .3, .3)
    assert {r['policy_id'] for r in nondominated(rows+[dominated])} == {r['policy_id'] for r in rows}
    for i, row in enumerate(rows):
        row['objective_loss'] = 1000*i
    assert len(select(rows, 3)) == 3


def test_depletion_is_a_real_third_objective():
    first = candidate(0, .8, .8, .4)
    second = candidate(.1, .8, .8, .2)
    tradeoff = candidate(.2, .7, .7, .1)
    assert {r['policy_id'] for r in nondominated([first, second, tradeoff])} == {second['policy_id'], tradeoff['policy_id']}


def test_infeasible_high_outcomes_do_not_replace_feasible_tradeoffs():
    feasible = candidate(0, .4, .4, .3)
    infeasible = candidate(.1, 1, 1, 0, False)
    assert select([feasible, infeasible], 3) == [feasible]
    assert select([infeasible], 3)[0]['search_feasible'] is False


def test_spread_preserves_each_objective_end_and_is_deterministic():
    rows = [candidate(i/10, 1-i/10, i/10, 1-abs(i-5)/10) for i in range(11)]
    selected = spread(rows, 4)
    for coordinate in range(3):
        assert min(values(r)[coordinate] for r in selected) == min(values(r)[coordinate] for r in rows)
    assert [r['policy_id'] for r in selected] == [r['policy_id'] for r in spread(rows[::-1], 4)]


def test_grid_refinement_keeps_endpoints_and_respects_budget():
    observed = []
    def evaluate(policy):
        x = policy.u
        observed.append(x)
        return candidate(x, .5+x/4, .5-x/4, .2)
    rows = grid_search(evaluate, [np.linspace(-.5, .5, 5)], lambda q: Policy(u=q[0]),
                       lambda p: abs(p.u) <= .5, 2, 3)
    assert min(observed) == -.5 and max(observed) == .5
    assert len(observed) > 5
    assert len(nondominated(rows)) == len(rows)


def test_main_aggregation_has_no_scalar_search_objective(tmp_path):
    config = load_config(ROOT/'configs/smoke.json')
    ctx = Evaluation(config, tmp_path, 100)
    result = ctx.aggregate(1, Policy(), 'train', 2)
    assert 'objective_loss' not in result
    np.testing.assert_array_equal(result['objective_vector'], [-result['mean_human_service'],
                                  -result['mean_ai_service'], result['mean_loss']])
    assert isinstance(ctx.objective(1, Policy()), list)


def test_more_conservative_main_test_budget_and_optional_scalar_controls():
    main = load_config(ROOT/'configs/research_protocol.json')
    assert main['objective']['name'] == 'pareto_human_ai_depletion'
    assert main['validation']['test_realizations'] == 128
    assert budget_plan(main)['simulations_started'] is False
    for name in ['human_only_priority', 'ai_only_priority']:
        assert load_config(ROOT/'configs'/f'{name}.json')['objective']['name'] != 'pareto_human_ai_depletion'


def test_report_separates_empirical_and_supported_frontiers(tmp_path, monkeypatch):
    config = load_config(ROOT/'configs/smoke.json')
    def assessed(u, h, a, loss, events=0):
        row = candidate(u, h, a, loss)
        row.update(count=128, human_services=[h]*128, ai_services=[a]*128, losses=[loss]*128,
                   stress_events=[True]*events+[False]*(128-events), stress_probability=events/128,
                   stress_probability_violation=events/128-.1)
        return row
    first, second = assessed(0, .9, .4, .2), assessed(.1, .4, .9, .2)
    uncertain = assessed(.2, .95, .95, .1, 8)
    reference = assessed(.3, .2, .2, .5)
    problem = dict(pressure=1, budget=.5, norm='L2')
    rows = [dict(problem_id='test', problem=problem,
                 families=dict(local=[first, second, uncertain], reference=[reference]))]
    assert assess_constraints(rows, .95, .1) == 4
    assert first['simultaneous_protection_supported']
    assert not uncertain['simultaneous_protection_supported']
    data = dict(problems=rows, primary_problem=problem, paired_outcome_comparisons=[], same_matrix_observation_controls=[])
    (tmp_path/'test_comparisons.json').write_text(json.dumps(data))
    destination = tmp_path/'analysis'
    destination.mkdir()
    monkeypatch.setattr(analysis, 'frontier_spatial_figures', lambda *args: None)
    analysis.analyze_frontiers(tmp_path, destination, dict(configuration=config, run_id='test'), [], [], 'Software check')
    with (destination/'pareto_frontier.csv').open() as stream:
        supported = [r for r in csv.DictReader(stream) if r['family'] == 'local']
    with (destination/'estimated_pareto_frontier.csv').open() as stream:
        empirical = [r for r in csv.DictReader(stream) if r['family'] == 'local']
    assert {r['policy_id'] for r in supported} == {first['policy_id'], second['policy_id']}
    assert {r['policy_id'] for r in empirical} == {uncertain['policy_id']}
    assert (destination/'pareto_frontier.png').stat().st_size > 1000
    assert (destination/'pareto_frontier.pdf').stat().st_size > 1000
