"""Paired world-level summaries, joint bounded intervals, family controls, and full inventories."""
import pathlib,json,ast,csv,concurrent.futures,collections,time
import numpy as np,pandas as pd
from scipy.special import logsumexp
from .runtime import P, OUT, O, WORKERS;B='db0eed394e20c822b2a6beb3'
def archived_interval_function():
 source=(P/'source_snapshot/src/human_ai_commons/metrics.py').read_text();tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='paired_simultaneous_intervals');space={'np':np,'logsumexp':logsumexp};exec(compile(ast.get_source_segment(source,node),str(P/'source_snapshot/src/human_ai_commons/metrics.py'),'exec'),space);return space['paired_simultaneous_intervals']


# bounded per-realization metric ranges. All new comparisons remain exploratory.
METRICS={
'first100_mean_cooperation':1,'formation_mean_cooperation':1,'late_mean_cooperation':1,
'first100_human_support':1,'first100_ai_support':1,'formation_human_support':1,'formation_ai_support':1,'full2000_human_support':1,'full2000_ai_support':1,'late_human_support':1,'late_ai_support':1,
'late_mean_stock':1,'late_stock_between_location_variance_share':1,'late_stock_pattern_first_last_correlation':2,'late_strategy_pattern_first_last_correlation':2,
'next_abundant_other_source_given_depleted_last_source':1,'late_preference_disagreement_fixed':1,'formation_preference_disagreement_fixed':1,
'firstcross_0.25_fraction_locations':1,'firstcross_0.5_fraction_locations':1,'firstcross_0.75_fraction_locations':1,'baseline_event_stock_after_100':1,'baseline_event_no_downcross_by_plus100':1,
'late_downcrossings_per_resource_step_0.5':1,'late_detrended_stock_self_corr_10':2,'late_detrended_stock_to_coop_1':2,
'late_cooperation_spectrum_power_5to9':1,'late_stock_cooperation_coherence_5to9':1,
'cooperation_moran':100,'local_cooperation_depleted_time_association':2,'human_worst10_agent_support':1,'ai_worst10_agent_support':1,'human_support_gini':1,'ai_support_gini':1,'persistent_cell_fraction':1,'full_window_stressed_resource_fraction':1}
ROBUST_METRICS=['mean_cooperation','cooperation_moran','local_cooperation_depleted_time_association','human_worst10_agent_support','ai_worst10_agent_support','human_support_gini','ai_support_gini','persistent_cell_fraction','full_window_stressed_resource_fraction','mean_accessible_resource']

def interval_chunk(task):
 specs,D,scales,K=task;n=D.shape[0];m=D.shape[1];confidence=1-.05*m/K
 simultaneous=archived_interval_function()(D/np.array(scales),confidence=confidence)
 rng=np.random.default_rng(20261113+n);weights=rng.multinomial(n,np.full(n,1/n),size=10000)/n;boots=weights@D;quant=np.quantile(boots,[.025,.975],axis=0)
 rows=[]
 for j,(meta,si,scale) in enumerate(zip(specs,simultaneous,scales)):
  rows.append(meta|{'paired_worlds':n,'difference':float(D[:,j].mean()),'standard_error':float(D[:,j].std(ddof=1)/np.sqrt(n)),'descriptive_lower':float(quant[0,j]),'descriptive_upper':float(quant[1,j]),'simultaneous_lower':si['lower']*scale,'simultaneous_upper':si['upper']*scale,'simultaneous_new_family_size':K,'new_family_confidence':.95,'status':'exploratory; familywise bounds for this listed family, not preregistered; original confirmatory intervals unchanged'})
 return rows

def main():
 start=time.time();a=pd.read_csv(O/'complete_realization_metrics.csv');s=pd.read_csv(O/'recovered_static_metrics.csv');e=pd.read_csv(O/'exact_immediate_effects.csv')
 s.groupby(['split','variant','pressure','policy_id'],dropna=False).mean(numeric_only=True).reset_index().to_csv(O/'static_condition_means.csv',index=False)
 keep=[c for c in s.columns if c not in a.columns or c=='evaluation_id'];a=a.merge(s[keep],on='evaluation_id',validate='one_to_one');a=a.merge(e,on=['evaluation_id','split','pressure','policy_id','index'],validate='one_to_one')
 for k in ['human','ai']:a['full2000_'+k+'_support']=(a['formation_'+k+'_support']+3*a['late_'+k+'_support'])/4
 a.to_csv(O/'integrated_trajectory_metrics.csv',index=False)
 means=a.groupby(['split','pressure','policy_id'],dropna=False).mean(numeric_only=True).reset_index();means.to_csv(O/'trajectory_condition_means.csv',index=False)
 meta=json.loads((P/'test_comparisons.json').read_text());candidates=pd.read_csv(P/'analysis/pareto_candidates.csv');testmeans=means[means.split=='test'].drop(columns='split');candidates.merge(testmeans,on=['pressure','policy_id'],suffixes=('','_trajectory')).to_csv(O/'all_tested_designs_with_mechanisms.csv',index=False)
 # combined frontiers are explicitly unconstrained sample means, retaining observation-rule distinctions.
 front=[]
 for pid,g in candidates.groupby('problem_id',sort=False):
  g=g.drop_duplicates('policy_id');v=np.column_stack([-g.human,-g.ai,g.depletion]);nd=~np.any(np.all(v[:,None,:]<=v[None,:,:],axis=2)&np.any(v[:,None,:]<v[None,:,:],axis=2),axis=0)
  for (_,r),flag in zip(g.iterrows(),nd):front.append({'problem_id':pid,'pressure':r.pressure,'budget':r.budget,'norm':r['norm'],'policy_id':r.policy_id,'unconstrained_combined_sample_frontier':bool(flag),'constraint_supported':bool(r.constraint_supported),'human':r.human,'ai':r.ai,'depletion':r.depletion})
 pd.DataFrame(front).to_csv(O/'combined_tested_frontiers.csv',index=False)
 # paired comparisons use complete world/driver indices; sites and steps are dependent.
 test=a[a.split=='test'];groups={(float(p),pid):g.set_index('index') for (p,pid),g in test.groupby(['pressure','policy_id'])};rob=s[s.split=='robustness'];rg={(v,pid):g.set_index('index') for (v,pid),g in rob.groupby(['variant','policy_id'])};specs=[];columns=[];scales=[]
 for (p,pid),g in groups.items():
  if pid==B:continue
  base=groups[(p,B)]
  for metric,scale in METRICS.items():
   d=(g[metric]-base[metric]).sort_index().to_numpy(float)
   if len(d)!=320 or not np.isfinite(d).all():continue
   specs.append({'comparison':'design_minus_original','condition':str(p),'pressure':p,'policy_id':pid,'reference_policy_id':B,'metric':metric,'scope':'test'});columns.append(d);scales.append(scale)
 for control in meta['same_matrix_observation_controls']:
  local=groups[(1,control['observations']['local']['policy_id'])]
  for obs in ['fixed','global']:
   other=groups[(1,control['observations'][obs]['policy_id'])]
   for metric,scale in METRICS.items():
    d=(local[metric]-other[metric]).sort_index().to_numpy(float)
    if len(d)!=320 or not np.isfinite(d).all():continue
    specs.append({'comparison':'same_matrix_local_minus_'+obs,'condition':'1','pressure':1,'policy_id':control['policy_id'],'reference_policy_id':control['observations'][obs]['policy_id'],'metric':metric,'scope':'test'});columns.append(d);scales.append(scale)
 r_specs=[];r_columns=[];r_scales=[]
 for (v,pid),g in rg.items():
  if pid==B:continue
  base=rg[(v,B)]
  for metric in ROBUST_METRICS:
   d=(g[metric]-base[metric]).sort_index().to_numpy(float)
   if len(d)!=256 or not np.isfinite(d).all():continue
   scale=400 if metric=='cooperation_moran' else (2 if 'association' in metric else 1)
   r_specs.append({'comparison':'frozen_transfer_minus_original','condition':v,'pressure':1,'policy_id':pid,'reference_policy_id':B,'metric':metric,'scope':'robustness'});r_columns.append(d);r_scales.append(scale)
 K=len(specs)+len(r_specs);jobs=[]
 for sp,cols,sc in [(specs,columns,scales),(r_specs,r_columns,r_scales)]:
  for j in range(0,len(sp),96):jobs.append((sp[j:j+96],np.column_stack(cols[j:j+96]),sc[j:j+96],K))
 print('new exploratory contrasts',K,'chunks',len(jobs),flush=True);allrows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
  for i,rs in enumerate(ex.map(interval_chunk,jobs),1):
   allrows+=rs
   if i%10==0:print('interval chunks',i,'/',len(jobs),'seconds',round(time.time()-start,1),flush=True)
 contrasts=pd.DataFrame(allrows);contrasts.to_csv(O/'all_exploratory_paired_comparisons.csv',index=False)
 # randomized one-action responses: whole-world descriptive means and bootstrap intervals, plus exact accounting check.
 causal=[]
 for (p,pid),g in groups.items():
  if pid!=B:continue
  cs=[c for c in g.columns if '_ipw_' in c or '_exact_' in c];validcols=[c for c in cs if np.isfinite(g[c].to_numpy(float)).all()];matrix=g.sort_index()[validcols].to_numpy(float)
  gen=np.random.default_rng(20261113);W=gen.multinomial(len(matrix),np.full(len(matrix),1/len(matrix)),size=10000)/len(matrix);quant=np.quantile(W@matrix,[.025,.975],axis=0)
  for j,c in enumerate(validcols):
   causal.append({'pressure':p,'metric':c,'mean':float(matrix[:,j].mean()),'descriptive_lower':float(quant[0,j]),'descriptive_upper':float(quant[1,j]),'independent_worlds':len(matrix),'scope':'one action changed, future policy natural; descriptive world bootstrap; not a trap-removal or all-cooperation intervention'})
 pd.DataFrame(causal).to_csv(O/'baseline_one_action_responses.csv',index=False)
 # paired link between early and late changes: descriptive correlations, never causal mediation.
 link=[]
 for (p,pid),g in groups.items():
  if pid==B:continue
  base=groups[(p,B)]
  for mech in ['first100_mean_cooperation','baseline_event_stock_after_100','firstcross_0.5_fraction_locations']:
   x=(g[mech]-base[mech]).sort_index().to_numpy()
   for outcome in ['late_human_support','late_ai_support','full2000_human_support','full2000_ai_support']:
    y=(g[outcome]-base[outcome]).sort_index().to_numpy();valid=np.isfinite(x)&np.isfinite(y);corr=float(np.corrcoef(x[valid],y[valid])[0,1]) if valid.sum()>2 and x[valid].std()>1e-15 and y[valid].std()>1e-15 else None
    link.append({'pressure':p,'policy_id':pid,'mechanism':mech,'activity':outcome,'correlation_of_paired_changes':corr,'scope':'descriptive association across matched worlds; no causal mediation claim'})
 pd.DataFrame(link).to_csv(O/'mechanism_activity_links.csv',index=False)
 # recovered cooperation compared with the archived spatial diagnostic.
 old=pd.read_csv(P/'analysis/spatial_characterization.csv',usecols=['evaluation_id','cooperation_spatial_autocorrelation']);check=s.merge(old,on='evaluation_id');moran_error=float(abs(check.cooperation_moran-check.cooperation_spatial_autocorrelation).max())
 audit={'detailed_trajectories':len(a),'test_trajectories':int((a.split=='test').sum()),'characterization_trajectories':int((a.split=='characterization').sum()),'distinct_test_conditions':len(groups),'static_records':len(s),'distinct_robustness_conditions':len(rg),'maximum_action_reconstruction_mismatches_nonboundary':int(a.nonboundary_action_mismatches.max()),'total_boundary_reconstruction_mismatches':int(a.reconstructed_action_mismatches.sum()),'maximum_source_balance_error':float(a.maximum_source_balance_error.max()),'maximum_recovered_spatial_moran_error':moran_error,'new_exploratory_contrasts':K,'new_familywise_resolved_contrasts':int(((contrasts.simultaneous_lower>0)|(contrasts.simultaneous_upper<0)).sum()),'elapsed_seconds':time.time()-start}
 (O/'completed_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit),flush=True)
if __name__=='__main__':main()
