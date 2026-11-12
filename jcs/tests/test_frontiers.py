from dataclasses import asdict
import csv
import json
import numpy as np
from human_ai_commons.frontiers import nondominated, select, spread, grid_search, values
from human_ai_commons.payoffs import Policy
from human_ai_commons.config import load_config, budget_plan, validate_config
from human_ai_commons.metrics import paired_simultaneous_intervals
from copy import deepcopy
import pytest
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


def test_equal_outcomes_are_not_mistaken_for_strict_dominance():
    first, second = candidate(0, .8, .7, .2), candidate(.1, .8, .7, .2)
    assert len(nondominated([first,second])) == 1
    assert {r['policy_id'] for r in nondominated([first,second],deduplicate=False)} == {first['policy_id'],second['policy_id']}


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
    assert main['validation']['test_realizations'] == 320
    plan = budget_plan(main)
    assert plan['simulations_started'] is False
    assert all(r['power'] >= .9 and r['zero_failure_upper_bound'] < .1 for r in plan['resource_assessment'].values())
    for name in ['human_only_priority', 'ai_only_priority']:
        assert load_config(ROOT/'configs'/f'{name}.json')['objective']['name'] != 'pareto_human_ai_depletion'


def test_resource_power_guard_rejects_an_inadequate_production_sample():
    config = deepcopy(load_config(ROOT/'configs/research_protocol.json'))
    config['validation']['robustness_realizations'] = 16
    with pytest.raises(ValueError,match='robustness sample'):
        validate_config(config)


def test_resource_power_plan_agrees_with_independent_exact_binomial_tests():
    from scipy.stats import binomtest, binom
    config = load_config(ROOT/'configs/research_protocol.json')
    for row in budget_plan(config)['resource_assessment'].values():
        threshold = (1-row['family_confidence'])/row['maximum_comparisons']
        n = row['realizations']
        accepted = [k for k in range(n+1) if binomtest(k,n,.1,alternative='less').pvalue <= threshold]
        assert max(accepted) == row['maximum_supported_failures']
        assert binom.cdf(max(accepted),n,.02) == pytest.approx(row['power'])


def test_resource_correction_covers_different_variants_of_the_same_policy():
    rows = []
    for variant in ['first','second']:
        row = candidate(0,.8,.8,.1)
        row['stress_events'] = [False]*256
        rows.append(dict(problem=dict(pressure=1),comparison_group=variant,families=dict(local=[row])))
    assert assess_constraints(rows,.95,.1) == 2
    assert all(r['families']['local'][0]['protected_family_size'] == 2 for r in rows)


def test_finite_sample_intervals_resolve_a_gain_with_small_between_world_variation():
    generator = np.random.default_rng(581)
    data = generator.uniform(0,.1,(320,60))
    result = paired_simultaneous_intervals(data,confidence=.9875)
    assert all(r['lower'] > 0 for r in result)
    assert all(r['bounded_lower'] < 0 for r in result)
    assert all(r['method'] == 'bounded_mean_betting_mixture' for r in result)


def test_simultaneous_intervals_preserve_sign_symmetry_and_row_order_invariance():
    data = np.random.default_rng(412).uniform(-.1,.1,(128,1))
    result = paired_simultaneous_intervals(np.column_stack([data,data,-data]))
    reordered = paired_simultaneous_intervals(np.column_stack([data,data,-data])[::-1])
    assert result[0]['lower'] == pytest.approx(result[1]['lower'])
    assert result[0]['lower'] == pytest.approx(-result[2]['upper'])
    assert result[0]['upper'] == pytest.approx(-result[2]['lower'])
    for first, second in zip(result,reordered):
        assert first['lower'] == pytest.approx(second['lower'])
        assert first['upper'] == pytest.approx(second['upper'])


def test_constant_observations_do_not_get_zero_width_statistical_intervals():
    result = paired_simultaneous_intervals(np.full((320,2),.005))
    assert all(r['lower'] < 0 < r['upper'] for r in result)
    assert all(-1 <= r['lower'] < r['upper'] <= 1 for r in result)


def test_finite_sample_intervals_match_independent_scalar_product_inversion():
    from scipy.optimize import brentq
    data = np.random.default_rng(691).uniform(-.4,.5,(64,2))
    result = paired_simultaneous_intervals(data)
    fractions = [.02,.05,.1,.2,.4,.6,.8,.95,1.]
    weights = [1/16]*8+[1/2]
    def scalar_bound(observations):
        def residual(mean):
            return sum(w*np.prod([1-rho+rho*x/mean for x in observations])
                       for w,rho in zip(weights,fractions))-80
        return brentq(residual,.001,float(np.mean(observations)),xtol=1e-13)
    for j,row in enumerate(result):
        normalized = (data[:,j]+1)/2
        assert row['lower'] == pytest.approx(2*scalar_bound(normalized)-1,abs=1e-11)
        assert row['upper'] == pytest.approx(1-2*scalar_bound(1-normalized),abs=1e-11)


@pytest.mark.parametrize('n',[16,32])
def test_exact_coverage_over_all_possible_bernoulli_sample_counts(n):
    from scipy.stats import binom
    intervals = [paired_simultaneous_intervals(np.array([1.]*k+[-1.]*(n-k))[:,None])[0]
                 for k in range(n+1)]
    for probability in np.linspace(0,1,201):
        true_mean = 2*probability-1
        misses = [true_mean < r['lower']-1e-12 or true_mean > r['upper']+1e-12 for r in intervals]
        error_probability = binom.pmf(np.arange(n+1),n,probability)[misses].sum()
        assert error_probability <= .05+1e-12


@pytest.mark.parametrize('data',[np.array([[0.],[np.nan]]),np.ones((2,1))*1.01,np.array([0.,.1])])
def test_finite_sample_intervals_reject_invalid_bounded_contrasts(data):
    with pytest.raises(ValueError):
        paired_simultaneous_intervals(data)


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
        supported = list(csv.DictReader(stream))
    with (destination/'estimated_pareto_frontier.csv').open() as stream:
        empirical = [r for r in csv.DictReader(stream) if r['family'] == 'local']
    assert {r['policy_id'] for r in supported} == {first['policy_id'], second['policy_id']}
    assert all(r['is_reference'] == 'False' for r in supported)
    assert {r['policy_id'] for r in empirical} == {uncertain['policy_id']}
    assert (destination/'pareto_frontier.png').stat().st_size > 1000
    assert (destination/'pareto_frontier.pdf').stat().st_size > 1000
    # An original game that dominates the tested alternatives remains an
    # available design in the actual family's frontier, not a self-only group.
    reference['mean_human_service'] = 1.
    reference['mean_ai_service'] = 1.
    reference['mean_loss'] = 0.
    reference['human_services'] = [1.]*128
    reference['ai_services'] = [1.]*128
    reference['losses'] = [0.]*128
    (tmp_path/'test_comparisons.json').write_text(json.dumps(data))
    analysis.analyze_frontiers(tmp_path, destination, dict(configuration=config, run_id='test'), [], [], 'Software check')
    with (destination/'pareto_frontier.csv').open() as stream:
        supported = list(csv.DictReader(stream))
    assert len(supported) == 1
    assert supported[0]['policy_id'] == reference['policy_id']
    assert supported[0]['family'] == 'local' and supported[0]['is_reference'] == 'True'
