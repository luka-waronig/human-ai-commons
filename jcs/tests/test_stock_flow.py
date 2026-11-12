import numpy as np
import pytest
from human_ai_commons.diagnostics import spatial_autocorrelation,spatial_summary
from human_ai_commons.dynamics import Scenario,Simulator,Drivers
from human_ai_commons.metrics import LocalStress,Service,StressSpec,WelfareSpec,request_fulfillment
from human_ai_commons.payoffs import Policy
from human_ai_commons.world import World,WorldSpec


@pytest.mark.parametrize('initial',[[.38,.38],[0.,.76]])
def test_constructed_low_reserve_state_can_supply_every_request(initial):
    # This is a mechanically valid constructed state, not a claim about typical runs.
    world=World(WorldSpec(width=10,height=1,humans=1,ai=1,capacity=1),
                [[0,0],[5,0]],[1.,1.],[.1,.1],[[0,0],[5,0]],[False,True],[False,False])
    scenario=Scenario(pressure=2/3,demand_ratio=1,theta=0,epsilon=0,allocation='proportional')
    sim=Simulator(world,scenario,Policy(forced='D'))
    sim.water[:]=initial
    stock=LocalStress(2,6,StressSpec(duration_steps=3))
    delivery=LocalStress(2,6,StressSpec(reference=1,duration_steps=3,basis='request_fulfillment'))
    service=Service(sim.cooperative_request*1.5,world.kinds,WelfareSpec())
    records=[[],[],[],[]]
    for step in range(6):
        event=sim.step(*Drivers(3,2).step(step))
        np.testing.assert_allclose(event['delivered'],event['requested'])
        np.testing.assert_allclose(sim.water,initial,atol=1e-14)
        stock.update(sim.water)
        delivery.update(request_fulfillment(event['delivered'],event['requested']))
        service.update(event['delivered'])
        for values,value in zip(records,[sim.water.copy(),sim.strategies.copy(),event['requested'],event['delivered']]):
            values.append(value)
    assert service.finish()['welfare']==pytest.approx(1)
    assert stock.finish()['persistent_local_stress']
    assert not delivery.finish()['persistent_local_stress']
    summary=spatial_summary(world,*[np.array(v) for v in records])
    assert summary['mean_accessible_stock']==pytest.approx(.38)
    assert summary['fraction_agent_steps_with_unmet_request']==0


def test_spatial_autocorrelation_distinguishes_clustering_and_alternation():
    adjacency=np.zeros((6,6))
    for i in range(6):
        adjacency[i,(i-1)%6]=adjacency[i,(i+1)%6]=1
    assert spatial_autocorrelation([0,0,0,1,1,1],adjacency)==pytest.approx(1/3)
    assert spatial_autocorrelation([0,1,0,1,0,1],adjacency)==pytest.approx(-1)
    assert spatial_autocorrelation([1]*6,adjacency) is None
