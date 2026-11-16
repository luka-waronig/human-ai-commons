"""Saved-state resource budgets: renewal, capacity clipping, withdrawals and storage.

These accounting calculations do not propagate counterfactual states or run the
commons model. Every term is evaluated along an already archived trajectory.
"""
import concurrent.futures
import csv
import hashlib
import json
import time
import numpy as np
from .complete_analysis import prepare, load_world
from .runtime import P, OUT, WORKERS


def process(task):
    split, pressure, pid, index, eid, path, world_path, _, policy = task
    world, spec = load_world(world_path)
    with np.load(P/path, allow_pickle=False) as saved:
        stocks = saved['stocks']
        strategy = saved['strategies'].astype(float)
        renewal = saved['resource_renewal']
        withdrawal = saved['source_withdrawal']
        accessible = saved['accessible']
        targets = saved['agent_targets']
    cap = world['capacity']
    previous_stock = np.concatenate([cap[None,:], stocks[:-1]])
    initial = np.full_like(world['initial'],policy['forced']=='C') if policy.get('forced') else world['initial']
    previous_action = np.concatenate([initial[None,:],strategy[:-1]])
    delta = abs(world['resources'][:,None,:]-world['positions'][None,:,:])
    distance = np.minimum(delta,np.array([spec['width'],spec['height']])-delta).max(axis=-1)
    membership = (distance <= spec['feedback_radius']).astype(float)
    counts = membership.sum(axis=1)
    composition = previous_action @ (membership/np.maximum(counts[:,None],1)).T
    n = previous_stock/cap
    factor = np.clip(1+.5*n*(1-n)*(3*composition-1),1,3)
    factor[:,counts==0] = 1
    potential = world['renewal']*factor
    unused = np.maximum(previous_stock+potential-cap,0)
    reconstructed = np.minimum(cap,previous_stock+potential)-previous_stock
    reconstruction_error = float(abs(reconstructed-renewal).max())
    if reconstruction_error > 1e-10:
        raise ValueError((eid,'renewal accounting mismatch',reconstruction_error))
    denominator = cap[accessible].sum()
    series = {'potential_renewal':potential[:,accessible].sum(axis=1)/denominator,
              'unused_renewal':unused[:,accessible].sum(axis=1)/denominator,
              'actual_renewal':renewal[:,accessible].sum(axis=1)/denominator,
              'withdrawal':withdrawal[:,accessible].sum(axis=1)/denominator}
    profile = {key:np.cumsum(values) for key,values in series.items()}
    profile['stock'] = stocks[:,accessible].sum(axis=1)/denominator
    balance = profile['stock']-1-profile['actual_renewal']+profile['withdrawal']
    potential_balance = profile['potential_renewal']-profile['unused_renewal']-profile['actual_renewal']
    if max(abs(balance).max(),abs(potential_balance).max()) > 1e-8:
        raise ValueError((eid,'cumulative accounting mismatch'))
    row = dict(split=split,pressure=pressure,policy_id=pid,index=index,evaluation_id=eid,
               accessible_sources=int(accessible.sum()),maximum_renewal_error=reconstruction_error,
               maximum_cumulative_balance_error=float(abs(balance).max()))
    row['target_history_sha256']=hashlib.sha256(targets.tobytes()).hexdigest()
    row['first_strategy_sha256']=hashlib.sha256(strategy[0].tobytes()).hexdigest()
    row['first_stock_sha256']=hashlib.sha256(stocks[0].tobytes()).hexdigest()
    selected=(stocks+withdrawal)/cap
    seen=selected[np.arange(len(targets))[:,None],np.maximum(targets,0)]
    for phase,begin,end in [('first100',0,100),('early',0,500),('late',500,len(stocks))]:
        for name,mask in [('human',~world['kinds']),('ai',world['kinds'])]:
            value=seen[begin:end,mask]
            row[f'{phase}_{name}_mean_selected_stock']=float(value.mean())
            row[f'{phase}_{name}_selected_below_half']=float((value<=.5).mean())
            row[f'{phase}_{name}_selected_below_all_switches']=float((value<1/21).mean())
            row[f'{phase}_{name}_selected_above_all_switches']=float((value>.75).mean())
    for step in (100,500,2000):
        for key,values in profile.items():row[f'{key}_step{step}'] = float(values[step-1])
    return row,profile


def main():
    tasks=prepare();rows=[];totals={};counts={};started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for j,(row,profile) in enumerate(pool.map(process,tasks,chunksize=4),1):
            rows.append(row);key=(row['split'],row['pressure'],row['policy_id'])
            if key not in totals:totals[key]={k:np.zeros_like(v) for k,v in profile.items()};counts[key]=0
            for k,v in profile.items():totals[key][k]+=v
            counts[key]+=1
            if j%1280==0:print(f'Resource budgets recovered: {j}/{len(tasks)} saved histories',flush=True)
    with (OUT/'resource_budget_worlds.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    profiles=[dict(split=key[0],pressure=key[1],policy_id=key[2],worlds=counts[key],
                   profiles={k:(v/counts[key]).tolist() for k,v in values.items()}) for key,values in totals.items()]
    (OUT/'resource_budget_profiles.json').write_text(json.dumps(profiles))
    audit=dict(histories=len(rows),maximum_renewal_error=max(r['maximum_renewal_error'] for r in rows),
               maximum_cumulative_balance_error=max(r['maximum_cumulative_balance_error'] for r in rows),
               elapsed_seconds=time.time()-started,scope='Exact accounting along saved histories; no model or counterfactual rollout')
    (OUT/'resource_budget_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)


if __name__=='__main__':
    main()
