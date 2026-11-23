"""Numbers behind the statements on what siting changes besides the averages, in the reference setting.

Reads the saved worlds of results/local_measures (configuration local_measures.json) and its per-agent
arrays; nothing is simulated. For each of the 64 worlds it compares random placement with siting of all AI
agents and prints:
- the share of the sources that human agents reach which AI agents also reach;
- the delivery limit from baseline renewal under both placements, and under siting with the sources left out
  that no agent reached under random placement, as the share of the rise in the limit that remains;
- the share of human agents whose share of demand met over the last 1000 of the compared steps is more than
  one percentage point larger, or smaller, under siting.
"""
import os
from dataclasses import replace
from pathlib import Path
import numpy as np
from human_ai_commons.paper_experiments import cell_context, delivery_ceiling, load_protocol
from human_ai_commons.paper_tables import array_file

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path(os.environ.get('JCS_RESULTS', ROOT/'results'))
config = load_protocol(ROOT/'configs/local_measures.json')
run = RESULTS/'local_measures'

overlap, rise, gain, loss = {'random': [], 'sited': []}, [], [], []
for index in range(config['worlds']):
    worlds = {}
    for name, cell in [('random', 'reference'), ('sited', 'sited_all')]:
        world, scenario, _, _ = cell_context(config, cell, index)
        worlds[name] = world
        humans = world.access[~world.kinds].any(axis=0)
        ais = world.access[world.kinds].any(axis=0)
        overlap[name].append((humans & ais).sum()/humans.sum())
    # requests keep the size they have under random placement, as in the run.
    random, sited = worlds['random'], worlds['sited']
    supply = random.renewal[random.access.any(axis=0)].sum()
    unit = scenario.pressure*supply/(random.spec.humans+scenario.demand_ratio*random.spec.ai)
    target = np.where(random.kinds, scenario.demand_ratio*unit, unit)*scenario.defect_ratio
    before, after = delivery_ceiling(random, target), delivery_ceiling(sited, target)
    # the sources that siting brings into reach for the first time give nothing in this variant.
    new = sited.access.any(axis=0) & ~random.access.any(axis=0)
    without_new = delivery_ceiling(replace(sited, renewal=np.where(new, 0., sited.renewal)), target)
    rise.append((without_new-before)/(after-before))
    with np.load(array_file(run, 'interventions', 'reference', index), allow_pickle=False) as saved:
        activity_random = saved['unchanged::final_activity']
    with np.load(array_file(run, 'interventions', 'sited_all', index), allow_pickle=False) as saved:
        activity_sited = saved['unchanged::final_activity']
    change = (activity_sited-activity_random)[~random.kinds]
    gain.append((change > .01).mean())
    loss.append((change < -.01).mean())

print(f"sources reached by human agents that AI agents also reach: {100*np.mean(overlap['random']):.1f}% under random placement, "
      f"{100*np.mean(overlap['sited']):.1f}% under siting")
print(f'share of the rise in the delivery limit that remains without newly reached sources: {100*np.mean(rise):.1f}% (mean over worlds)')
print(f'human agents with more than one percentage point more of their demand met under siting: {100*np.mean(gain):.1f}%, '
      f'less: {100*np.mean(loss):.1f}%')
