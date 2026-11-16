"""Complete accounting and inference checks recovered from the completed study.

Additional analyses are exploratory. Original planned confidence intervals and
simulation records are never replaced. No model simulation is executed.
"""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .aggregate_all import archived_interval_function
from .runtime import P, OUT

REFERENCE='db0eed394e20c822b2a6beb3'


def coverage_components(human, ai, human_cooperation, ai_cooperation):
    """Exact linear target-coverage gap: lower requests plus unfilled requests."""
    human_requested=1-np.asarray(human_cooperation)/3
    ai_requested=1-np.asarray(ai_cooperation)/3
    return dict(human_requested=human_requested,ai_requested=ai_requested,
                human_unfilled=human_requested-np.asarray(human),
                ai_unfilled=ai_requested-np.asarray(ai),
                lower_request_contribution=(np.asarray(ai_cooperation)-np.asarray(human_cooperation))/3,
                unfilled_request_contribution=(ai_requested-np.asarray(ai))-(human_requested-np.asarray(human)),
                coverage_gap=np.asarray(human)-np.asarray(ai))


def planned_matrix(problems, robustness=False):
    columns={};metadata={}
    for g in problems:
        pressure=g['problem']['pressure'];condition=g['variant']['name'] if robustness else pressure
        baseline=g['families']['reference'][0]
        for family,alternatives in g['families'].items():
            for candidate in alternatives:
                if candidate['policy_id']==REFERENCE:continue
                assert candidate['world_ids']==baseline['world_ids']
                for outcome,field in [('human','human_services'),('ai','ai_services'),('depletion','losses')]:
                    key=(str(condition),candidate['policy_id'],outcome)
                    values=np.asarray(candidate[field])-np.asarray(baseline[field])
                    if key in columns:assert np.array_equal(columns[key],values)
                    columns[key]=values
                    metadata[key]=dict(scope='robustness' if robustness else 'main',condition=condition,
                                       policy_id=candidate['policy_id'],outcome=outcome)
    keys=sorted(columns)
    return np.column_stack([columns[k] for k in keys]),[metadata[k] for k in keys]


def bootstrap_sensitivity(blocks, samples=10000):
    """Approximate simultaneous max-standardized-mean bands across both splits.

    Each block uses whole matched world indices; main and robustness splits
    are resampled independently. These do not have the original finite-sample guarantee.
    """
    generator=np.random.default_rng(20261114);maximum=np.zeros(samples);results=[]
    for matrix,metadata in blocks:
        n=matrix.shape[0];se=matrix.std(axis=0,ddof=1)/np.sqrt(n);mean=matrix.mean(axis=0)
        weights=generator.multinomial(n,np.full(n,1/n),size=samples)/n-1/n
        for j in range(0,matrix.shape[1],64):
            z=abs((weights@matrix[:,j:j+64])/np.where(se[j:j+64]>1e-14,se[j:j+64],np.inf))
            maximum=np.maximum(maximum,z.max(axis=1))
        results.append((metadata,mean,se))
    critical=float(np.quantile(maximum,.95));rows=[]
    for metadata,mean,se in results:
        for i,m in enumerate(metadata):
            rows.append(m|dict(difference=mean[i],standard_error=se[i],
                               exploratory_joint_lower=mean[i]-critical*se[i],
                               exploratory_joint_upper=mean[i]+critical*se[i],
                               critical_value=critical,bootstrap_replicates=samples,
                               method='Exploratory approximate whole-world max-t bootstrap across all original outcome contrasts; original planned inference unchanged'))
    return pd.DataFrame(rows)


def main():
    static=pd.read_csv(OUT/'recovered_static_metrics.csv')
    fields=coverage_components(static.human_service,static.ai_service,
                               static.human_mean_cooperation,static.ai_mean_cooperation)
    accounting=static[['evaluation_id','split','variant','pressure','policy_id','index','human_service','ai_service','human_mean_cooperation','ai_mean_cooperation']].copy()
    for name,values in fields.items():accounting[name]=values
    error=float(abs(accounting.coverage_gap-accounting.lower_request_contribution-accounting.unfilled_request_contribution).max())
    assert error<1e-12,error
    accounting.to_csv(OUT/'coverage_accounting_worlds.csv',index=False)
    accounting.groupby(['split','variant','pressure','policy_id'],dropna=False).mean(numeric_only=True).reset_index().to_csv(OUT/'coverage_accounting_means.csv',index=False)
    budgets=pd.read_csv(OUT/'resource_budget_worlds.csv');means=[]
    for (split,pressure,pid),g in budgets.groupby(['split','pressure','policy_id']):
        baseline=budgets[(budgets.split==split)&(budgets.pressure==pressure)&(budgets.policy_id==REFERENCE)].set_index('index')
        v=g.set_index('index')
        assert set(v.index)==set(baseline.index)
        for step in (100,500,2000):
            d={name:float((v[f'{name}_step{step}']-baseline[f'{name}_step{step}']).mean()) for name in ['stock','potential_renewal','unused_renewal','actual_renewal','withdrawal']}
            assert abs(d['stock']-d['potential_renewal']+d['unused_renewal']+d['withdrawal'])<1e-10
            means.append(dict(split=split,pressure=pressure,policy_id=pid,step=step,worlds=len(g),**d))
    pd.DataFrame(means).to_csv(OUT/'paired_resource_budget_means.csv',index=False)
    recorded=budgets[budgets.split=='test']
    routing_identical=bool((recorded.groupby('index').target_history_sha256.nunique()==1).all())
    first_identical=bool((recorded.groupby(['pressure','index'])[['first_strategy_sha256','first_stock_sha256']].nunique()==1).all().all())
    assert routing_identical and first_identical
    test=json.loads((P/'test_comparisons.json').read_text());rob=json.loads((P/'robustness.json').read_text())
    blocks=[planned_matrix(test['problems']),planned_matrix(rob,True)]
    sensitivity=bootstrap_sensitivity(blocks);sensitivity.to_csv(OUT/'outcome_inference_sensitivity.csv',index=False)
    # interval widths for an all-zero observed contrast vector.
    radii=[]
    for n,k,label in [(320,309,'main'),(256,585,'robustness')]:
        interval=archived_interval_function()(np.zeros((n,1)),confidence=1-.0125/k)[0]
        radii.append(dict(scope=label,worlds=n,contrasts=k,original_family_error=.0125,
                          all_zero_lower=interval['lower'],all_zero_upper=interval['upper']))
    pd.DataFrame(radii).to_csv(OUT/'original_interval_resolution.csv',index=False)
    # full sources verify the paired driver identities and all recorded initial-state outputs.
    initial_hashes={};drivers={};reference_ids={};source_events={}
    for line in (P/'realizations.jsonl').open():
        record=json.loads(line)
        if record['split']=='test':
            source_events[(record['scenario']['pressure'],record['policy_id'],record['index'])]=np.asarray(record['site_longest_continuous_stress_steps'])>=150
        if record['split']!='robustness':continue
        key=(record['variant'],record['policy_id'],record['index'])
        drivers[key]=record['driver_seed']
        if record['policy_id']==REFERENCE:reference_ids[key]=record['world_id']
        if record['variant'] in ['initial_0.1','initial_0.9']:
            clean={k:v for k,v in record.items() if k not in ['world_id','world_path','evaluation_id','variant']}
            initial_hashes[key]=hashlib.sha256(json.dumps(clean,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    paired=[]
    for (variant,pid,index),value in initial_hashes.items():
        if variant!='initial_0.1':continue
        other=('initial_0.9',pid,index)
        paired.append(value==initial_hashes[other]);assert drivers[(variant,pid,index)]==drivers[other]
    assert all(paired)
    spatial=[]
    for (pressure,pid,index),event in source_events.items():
        baseline=source_events[(pressure,REFERENCE,index)]
        assert event.shape==baseline.shape
        relieved=float((baseline&~event).mean());newly=float((~baseline&event).mean())
        assert abs(event.mean()-baseline.mean()-newly+relieved)<1e-12
        spatial.append(dict(pressure=pressure,policy_id=pid,index=index,accessible_sources=len(event),
                            original_affected_fraction=float(baseline.mean()),design_affected_fraction=float(event.mean()),
                            relieved_source_fraction=relieved,newly_affected_source_fraction=newly,
                            retained_affected_fraction=float((baseline&event).mean()),
                            any_affected=bool(event.any())))
    spatial=pd.DataFrame(spatial);spatial.to_csv(OUT/'paired_source_events_worlds.csv',index=False)
    spatial.groupby(['pressure','policy_id']).mean(numeric_only=True).reset_index().to_csv(OUT/'paired_source_events_means.csv',index=False)
    # both initial-state variants are default-equivalent under initially full sources.
    rb=static[static.split=='robustness'];default=rb[rb.variant=='initial_0.1'].set_index(['policy_id','index'])
    scenario_columns=[];scenario_metadata=[];interactions=[]
    for variant,g in rb.groupby('variant'):
        if variant=='initial_0.1':continue
        original=g[g.policy_id==REFERENCE].set_index('index')
        base=default.loc[REFERENCE]
        for metric in ['human_service','ai_service','mean_accessible_resource','persistent_cell_fraction','mean_cooperation']:
            difference=(original[metric]-base[metric]).sort_index().to_numpy()
            assert len(difference)==256
            scenario_columns.append(difference);scenario_metadata.append(dict(variant=variant,metric=metric,difference=float(difference.mean())))
        for pid,v in g.groupby('policy_id'):
            if pid==REFERENCE:continue
            v=v.set_index('index');same=default.loc[pid]
            for metric in ['human_service','ai_service','loss']:
                change=(v[metric]-original[metric])-(same[metric]-base[metric])
                interactions.append(dict(variant=variant,policy_id=pid,outcome=metric,
                                         paired_interaction_mean=float(change.mean()),worlds=256,
                                         scope='Descriptive paired change in the payoff effect under the variant versus the default-equivalent initial-state control'))
    matrix=np.column_stack(scenario_columns)
    bounds=archived_interval_function()(matrix,confidence=.95)
    scenario=[]
    for m,interval in zip(scenario_metadata,bounds):scenario.append(m|{'exploratory_simultaneous_lower':interval['lower'],'exploratory_simultaneous_upper':interval['upper'],'paired_worlds':256,'family_size':len(bounds),'scope':'Exploratory additional bounded-mean family; original comparisons unchanged'})
    pd.DataFrame(scenario).to_csv(OUT/'paired_assumption_effects.csv',index=False)
    pd.DataFrame(interactions).to_csv(OUT/'payoff_assumption_interactions.csv',index=False)
    # the default-equivalent initial-state ensemble is independent of the main test split.
    primary_ids=set(pd.read_csv(P/'analysis/pareto_candidates.csv').query('pressure==1 and budget==0.5 and norm=="L2"').policy_id)
    main_default=static[(static.split=='test')&(static.pressure==1)&static.policy_id.isin(primary_ids)]
    repeat=[]
    for pid in default.index.get_level_values('policy_id').unique():
        if pid==REFERENCE or pid not in primary_ids:continue
        for metric in ['human_service','ai_service','loss']:
            first=(main_default[main_default.policy_id==pid].set_index('index')[metric]-main_default[main_default.policy_id==REFERENCE].set_index('index')[metric]).to_numpy()
            second=(default.loc[pid][metric]-default.loc[REFERENCE][metric]).to_numpy()
            repeat.append(dict(policy_id=pid,outcome=metric,main_worlds=len(first),repeat_worlds=len(second),
                               main_difference=float(first.mean()),repeat_difference=float(second.mean()),
                               pooled_difference=float(np.concatenate([first,second]).mean()),
                               same_mean_direction=bool(first.mean()*second.mean()>0),
                               scope='Descriptive independent default-equivalent repeat; two identical initial-state variants count once; original inference unchanged'))
    pd.DataFrame(repeat).to_csv(OUT/'default_model_repeatability.csv',index=False)
    # paired change distributions under the shared-random-input coupling.
    main=static[static.split=='test'];distribution=[]
    for (pressure,pid),g in main.groupby(['pressure','policy_id']):
        if pid==REFERENCE:continue
        v=g.set_index('index');base=main[(main.pressure==pressure)&(main.policy_id==REFERENCE)].set_index('index')
        h=v.human_service-base.human_service;a=v.ai_service-base.ai_service
        distribution.append(dict(pressure=pressure,policy_id=pid,worlds=len(v),
            human_mean=h.mean(),ai_mean=a.mean(),human_q10=h.quantile(.1),human_median=h.median(),human_q90=h.quantile(.9),
            ai_q10=a.quantile(.1),ai_median=a.median(),ai_q90=a.quantile(.9),
            both_gain_fraction=((h>1e-12)&(a>1e-12)).mean(),both_lose_fraction=((h<-1e-12)&(a<-1e-12)).mean(),
            scope='Descriptive distribution under the study matched-world coupling, not a coupling-independent superiority probability'))
    pd.DataFrame(distribution).to_csv(OUT/'paired_activity_distributions.csv',index=False)
    audit=dict(coverage_accounting_rows=len(accounting),routing_policy_and_pressure_invariant=routing_identical,first_record_policy_invariant=first_identical,maximum_coverage_identity_error=error,
               initial_state_matched_pairs=len(paired),initial_state_exact_record_matches=sum(paired),
               main_outcome_contrasts=blocks[0][0].shape[1],robustness_outcome_contrasts=blocks[1][0].shape[1],
               approximate_joint_resolved=int(((sensitivity.exploratory_joint_lower>0)|(sensitivity.exploratory_joint_upper<0)).sum()),
               bootstrap_critical_value=float(sensitivity.critical_value.iloc[0]),
               paired_assumption_effects=len(scenario),scope='Additional saved-data audit; all new inference is exploratory; original intervals retained')
    (OUT/'additional_evidence_audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)


if __name__=='__main__':
    main()
