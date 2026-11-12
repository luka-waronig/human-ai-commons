from dataclasses import replace
import numpy as np
import pytest
from human_ai_commons.capacity import routing_bound, strict_safety_bound
from human_ai_commons.dynamics import Drivers, Scenario, Simulator, simulate
from human_ai_commons.metrics import Depletion, MetricSpec, rescore, upper_tail_mean
from human_ai_commons.optimization import differential_search
from human_ai_commons.payoffs import Policy, resource_only_encoding, effective_change_interval
from human_ai_commons.world import World, WorldSpec, generate_world, torus_distances, transform_world


def hand_world():
    spec = WorldSpec(width=2, height=1, humans=2, ai=0, density=1, capacity=1)
    return World(spec, np.array([[0, 0]]), np.array([1.]), np.array([.1]),
                 np.array([[0, 0], [1, 0]]), np.array([False, False]), np.array([True, False]))


def test_original_payoffs_and_decisions_without_legacy_dependency():
    for n in np.linspace(0, 1, 21):
        old = (1-n)*np.array([[5, 1], [3, 0]])+n*np.array([[3, 0], [5, 1]])
        np.testing.assert_allclose(2*Policy().matrix(n)+1, old, atol=1e-15)
        for x in np.linspace(0, 1, 11):
            assert (Policy().gap(n, x) <= 0) == (n <= .5)


@pytest.mark.parametrize('t,s', [(2., -.5), (3., -1.), (1.5, -2.)])
@pytest.mark.parametrize('norm', ['L1', 'L2'])
def test_resource_only_encoding_cost_and_box(t, s, norm):
    a,b=t-1,-s
    for k in [-.95, -.5, 0, .5, 2.]:
        p=resource_only_encoding(k,t,s,(-.95,2.),norm)
        assert np.all(p.relative >= -.95-1e-14) and np.all(p.relative <= 2+1e-14)
        assert (p.u+p.v)/(a+b) == pytest.approx(k)
        assert p.cost(norm) <= Policy.from_relative(k,k,t,s).cost(norm)+1e-14
        for r in np.linspace(-.95,2,501):
            other=((a+b)*k-a*r)/b
            if -.95 <= other <= 2:
                assert p.cost(norm) <= Policy.from_relative(r,other,t,s).cost(norm)+1e-14
    for k in effective_change_interval(.5,t,s,norm):
        assert resource_only_encoding(k,t,s,(-.95,2.),norm).cost(norm) <= .5+1e-14


def test_shared_budget_original_zero_and_equal_relative_units():
    assert Policy().cost() == 0
    assert Policy.from_relative(.5,.5).cost() == pytest.approx(.5)
    assert resource_only_encoding(.5).cost() < .5


def test_simultaneous_observations_are_not_contaminated_by_agent_order():
    world = hand_world()
    policy = Policy.from_relative(1., -.5)
    sync = Simulator(world, Scenario(pressure=.1, theta=0, epsilon=0), policy)
    sequential = Simulator(world, Scenario(pressure=.1, theta=0, epsilon=0, schedule='sequential_planning'), policy)
    for sim in [sync, sequential]:
        sim.water[:] = .4  # Replenishment gives exactly n=0.5 before both agents plan.
        sim.step(np.array([.1, .1]), np.array([.5, .5]), np.array([0, 1]))
    np.testing.assert_array_equal(sync.strategies, [False, True])
    np.testing.assert_array_equal(sequential.strategies, [False, False])


def test_conservation_accounting_and_unmet_requests():
    world = hand_world()
    simulator = Simulator(world, Scenario(pressure=20), Policy(forced='C'))
    for t in range(12):
        before = simulator.water.sum()
        event = simulator.step(*Drivers(33, 2).step(t))
        assert simulator.water.sum() == pytest.approx(before+event['actual_renewal']-event['delivered'].sum())
        assert np.all(event['delivered'] <= event['requested'])
        assert np.all(simulator.water >= 0)
        assert np.all(simulator.water <= world.capacity)
    assert event['delivered'].sum() < event['requested'].sum()


def test_world_replay_immutability_and_pressure_normalization(tmp_path):
    world = generate_world(WorldSpec(width=4, height=5, humans=4, ai=4, density=.5), 15)
    world.save(tmp_path/'world.npz')
    restored = World.load(tmp_path/'world.npz')
    assert world.id == restored.id
    with pytest.raises(ValueError):
        world.renewal[0] = 0
    for ratio in [5, 20, 40]:
        sim = Simulator(world, Scenario(pressure=1.3, demand_ratio=ratio), Policy())
        assert sim.cooperative_request.sum() == pytest.approx(1.3*world.renewal[sim.accessible].sum())
    clustered = transform_world(world, cluster_ai=True)
    np.testing.assert_array_equal(world.positions, clustered.positions)
    assert clustered.kinds.sum() == world.kinds.sum()
    assert clustered.access.any(axis=1).all()


def test_dense_rectangular_world_and_small_torus_deduplicate_neighbors():
    world = generate_world(WorldSpec(width=2, height=3, humans=9, ai=9, density=1), 44)
    _, counts = np.unique(world.positions, axis=0, return_counts=True)
    assert counts.max() == 3
    simulator = Simulator(world, Scenario(), Policy())
    np.testing.assert_array_equal(simulator.ecological_count, np.full(6, 18))
    assert torus_distances(np.array([[0, 0]]), np.array([[1, 2]]), 2, 3)[0, 0] == 1


def test_indexed_random_drivers_and_recording_do_not_change_dynamics():
    world = generate_world(WorldSpec(width=4, height=4, humans=2, ai=2, density=.5), 19)
    drivers = Drivers(15, 4)
    first = drivers.step(3)
    drivers.step(100)
    for a, b in zip(first, drivers.step(3)):
        np.testing.assert_array_equal(a, b)
    a, _ = simulate(world, Scenario(), Policy(), 15, 15, 3, record=False)
    b, data = simulate(world, Scenario(), Policy(), 15, 15, 3, record=True)
    assert a == b
    assert rescore(data['stocks'], world.capacity, data['accessible'], 3, MetricSpec())['loss'] == a['loss']


def test_fractional_tail_and_empty_cells_remain_in_score():
    assert upper_tail_mean([0, .2, .6, 1], .375) == pytest.approx((1+.5*.6)/1.5)
    tracker = Depletion(2, MetricSpec())
    for _ in range(5):
        tracker.update([0, 1])
    assert tracker.finish()['loss'] == 1
    assert tracker.finish()['longest_continuous_depletion_steps'] == 5
    tracker.update([1, 1])
    assert tracker.finish()['loss'] == pytest.approx(5/6)


def test_physical_cut_can_fail_with_total_capacity_surplus():
    restricted = routing_bound([[True, False], [False, True]], [20, 2], [13, 13])
    assert restricted['total_capacity'] > restricted['required']
    assert restricted['relaxed_routing_infeasible']
    assert restricted['cut_demand'] > restricted['cut_capacity']
    generous = routing_bound(np.ones((2, 2), bool), [20, 2], [13, 13])
    assert not generous['relaxed_routing_infeasible']
    assert generous['relaxed_deliverable'] == pytest.approx(22)
    isolated = routing_bound([[False]], [2], [3])
    assert isolated['relaxed_routing_infeasible']


def test_isolated_agent_does_not_falsely_certify_resource_depletion():
    spec = WorldSpec(width=10, height=1, humans=2, ai=0, require_access=False)
    world = World(spec, [[0, 0]], [1], [.1], [[0, 0], [5, 0]], [False, False], [True, True])
    result = strict_safety_bound(world, Scenario(pressure=.01), 10, .2)
    assert result['excluded_isolated_agents'] == 1
    assert not result['strict_safety_ruled_out']


def test_optimizer_on_constrained_analytic_objective():
    settings = dict(explicit_population_size=8, maximum_generations=5, strategy='best1bin',
                    mutation=[.5, 1], recombination=.7)
    def run():
        points = []
        def evaluate(policy):
            assert policy.cost() <= .5+1e-12
            value = (policy.q_c-.8)**2+(policy.q_d-.3)**2
            points.append((policy.q_c, policy.q_d, value))
            return value
        differential_search(evaluate, .5, 2., -.5, 'L2', (-.95, 2.), settings, 123, Policy.from_relative(.25, .25))
        return points
    a, b = run(), run()
    assert a == b
    assert min(r[2] for r in a) < (.5-.8)**2+(.5-.3)**2


@pytest.mark.parametrize('kwargs', [dict(width=0), dict(density=0), dict(humans=-1), dict(capacity=np.nan)])
def test_invalid_worlds_rejected(kwargs):
    with pytest.raises(ValueError):
        WorldSpec(**kwargs)
