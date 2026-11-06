import numpy as np
import pytest
from watertoc_jcs.payoffs import Policy
from watertoc_jcs.optimization import one_dimensional


@pytest.mark.parametrize('u,v,qc,qd', [(0,0,.5,.5), (1,.5,2/3,2/3), (1,0,2/3,.5), (0,.5,.5,2/3)])
def test_transition_levels_are_derived_from_the_matrix(u,v,qc,qd):
    policy=Policy(u,v)
    assert policy.q_c == pytest.approx(qc)
    assert policy.q_d == pytest.approx(qd)
    assert policy.gap(qc,1) == pytest.approx(0,abs=1e-15)
    assert policy.gap(qd,0) == pytest.approx(0,abs=1e-15)
    for n in np.linspace(0,1,9):
        expected=(1-n)*np.array([[2+u,v],[1,-.5]])+n*np.array([[1,-.5],[2,0]])
        np.testing.assert_allclose(policy.matrix(n),expected)
        for x in [0,.2,.5,.8,1]:
            advantages=expected[1]-expected[0]
            assert policy.gap(n,x) == pytest.approx(x*advantages[0]+(1-x)*advantages[1])


def test_unequal_changes_do_not_incorrectly_keep_the_slope_ratio_fixed():
    assert Policy(1,0).omega == pytest.approx(1/3)
    assert Policy(0,.5).omega == pytest.approx(3/4)


def test_fixed_environment_synchronous_response_matches_the_game_type():
    def response(policy,x):
        return .9 if policy.gap(.6,x)<=0 else .1
    matching,opposite=Policy(1,0),Policy(0,.5)
    assert response(matching,.1)==.1 and response(matching,.9)==.9
    assert response(opposite,.1)==.9 and response(opposite,.9)==.1


def test_redistribution_preserves_the_original_rule_but_changes_local_response():
    original,changed=Policy(),Policy(.25,-.25)
    for n in np.linspace(0,1,21):
        np.testing.assert_allclose(original.matrix(n).sum(axis=1),changed.matrix(n).sum(axis=1))
        assert original.gap(n,.5) == pytest.approx(changed.gap(n,.5))
    assert changed.gap(.5,0)>0
    assert changed.gap(.5,1)<0


@pytest.mark.parametrize('norm',['L1','L2'])
def test_redistribution_search_stays_inside_the_same_budget_and_preserves_row_sums(norm):
    seen=[]
    def evaluate(p):
        assert p.u+p.v == 0
        assert p.cost(norm) <= .5+1e-12
        assert np.all(p.relative >= -.95) and np.all(p.relative <= 2)
        seen.append(p)
        return -p.u
    settings=dict(initial_grid_points=5,refinement_rounds=1,refined_intervals_per_round=1,refinement_grid_points=3)
    one_dimensional(evaluate,.5,2,-.5,norm,settings,(-.95,2),'redistribution')
    assert len(seen)>3


@pytest.mark.parametrize('kwargs',[dict(u=-1),dict(v=-.5),dict(t=1),dict(s=0),dict(u=np.nan)])
def test_depleted_cooperation_and_replete_defection_conditions_are_enforced(kwargs):
    with pytest.raises(ValueError):
        Policy(**kwargs)
