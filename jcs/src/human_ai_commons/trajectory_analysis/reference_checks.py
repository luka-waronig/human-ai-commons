"""Reference-world checks for request fulfilment, threshold exposure and averaging order."""
import concurrent.futures
import csv
import json
import numpy as np
from .complete_analysis import prepare, load_world
from .runtime import P, OUT, WORKERS


def process(task):
    _, pressure, _, index, evaluation_id, trajectory, world_path, *_ = task
    world, _ = load_world(world_path)
    with np.load(P/trajectory, allow_pickle=False) as data:
        n = data['stocks'][500:]/world['capacity']
        accessible = data['accessible']
        requests = data['agent_requested'][500:]
        delivered = data['agent_delivered'][500:]
        targets = data['agent_targets'][500:]
        replenished = (data['stocks'][500:]+data['source_withdrawal'][500:])/world['capacity']
    kinds = world['kinds']
    valid = targets >= 0
    seen = replenished[np.arange(len(targets))[:,None], np.maximum(targets,0)][valid]
    deficit = np.maximum(.2-n[:,accessible],0)/.2
    # fractional upper-tail weights; this reverses the executed time-then-space order.
    ordered = np.sort(deficit, axis=1)[:,::-1]
    tail_size = .1*ordered.shape[1]
    whole = int(tail_size)
    reverse = (ordered[:,:whole].sum(axis=1)+(tail_size-whole)*ordered[:,whole])/tail_size
    shortage = delivered < requests-1e-12
    row = dict(evaluation_id=evaluation_id, pressure=pressure, index=index,
               request_fulfilment=float(np.minimum(delivered/requests,1).mean()),
               shortage_agent_step_fraction=float(shortage.mean()),
               any_agent_uninterrupted_shortage=bool(shortage.all(axis=0).any()),
               any_source_below_point2_entire_window=bool((n[:,accessible]<.2).all(axis=0).any()),
               selected_stock_below_all_switches=float((seen<1/21).mean()),
               selected_stock_above_all_switches=float((seen>.75).mean()),
               reversed_averaging_depletion=float(reverse.mean()))
    for name, mask in [('human',~kinds),('ai',kinds)]:
        row[name+'_request_fulfilment'] = float(np.minimum(delivered[:,mask]/requests[:,mask],1).mean())
    return row


def main():
    tasks = [t for t in prepare() if t[0]=='test' and t[2]=='db0eed394e20c822b2a6beb3']
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        rows = list(pool.map(process,tasks,chunksize=4))
    with (OUT/'reference_request_and_threshold_checks.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(f'Reference request and threshold checks: {len(rows)} saved worlds',flush=True)


if __name__=='__main__':
    main()
