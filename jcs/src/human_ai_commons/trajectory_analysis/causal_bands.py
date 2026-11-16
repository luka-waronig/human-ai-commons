"""Exploratory world-cluster bootstrap bands for all one-action responses and contrasts."""
import pathlib,numpy as np,pandas as pd,json,time
from .runtime import P, OUT, O, WORKERS;B='db0eed394e20c822b2a6beb3'
def main():
 start=time.time();a=pd.read_csv(O/'integrated_trajectory_metrics.csv',low_memory=False);a=a[a.split=='test'];groups={(float(p),pid):g.set_index('index').sort_index() for (p,pid),g in a.groupby(['pressure','policy_id'])};metrics=[c for c in a.columns if '_ipw_' in c or '_exact_' in c];specs=[];columns=[]
 for (p,pid),g in groups.items():
  for c in metrics:
   v=g[c].to_numpy(float)
   if not np.isfinite(v).all():continue
   specs.append({'comparison':'one_action_cooperate_minus_defect','pressure':p,'policy_id':pid,'reference_policy_id':'one_action_defection','metric':c});columns.append(v)
  for c in [c for c in metrics if '_ipw_neighborhood_cooperation_' in c or '_ipw_source_stock_' in c or '_ipw_future_sensed_stock_' in c]:
   if pid==B:continue
   v=(g[c]-groups[(p,B)][c]).to_numpy(float);specs.append({'comparison':'one_action_response_design_minus_original','pressure':p,'policy_id':pid,'reference_policy_id':B,'metric':c});columns.append(v)
 control=json.loads((P/'test_comparisons.json').read_text())['same_matrix_observation_controls']
 for c in control:
  local=groups[(1,c['observations']['local']['policy_id'])]
  for obs in ['fixed','global']:
   other=groups[(1,c['observations'][obs]['policy_id'])]
   for metric in [c for c in metrics if '_ipw_neighborhood_cooperation_' in c or '_ipw_source_stock_' in c or '_ipw_future_sensed_stock_' in c]:
    v=(local[metric]-other[metric]).to_numpy(float);specs.append({'comparison':'same_matrix_response_local_minus_'+obs,'pressure':1,'policy_id':c['policy_id'],'reference_policy_id':c['observations'][obs]['policy_id'],'metric':metric});columns.append(v)
 D=np.column_stack(columns);mu=D.mean(axis=0);se=D.std(axis=0,ddof=1)/np.sqrt(320);gen=np.random.default_rng(20261113);W=gen.multinomial(320,np.full(320,1/320),size=5000)/320;centered=W-1/320;maxz=np.zeros(len(W));low=np.empty(len(specs));high=np.empty(len(specs))
 for j in range(0,D.shape[1],128):
  v=D[:,j:j+128];boot=W@v;low[j:j+128],high[j:j+128]=np.quantile(boot,[.025,.975],axis=0);z=abs((centered@v)/np.where(se[j:j+128]>1e-14,se[j:j+128],np.inf));maxz=np.maximum(maxz,z.max(axis=1))
 critical=float(np.quantile(maxz,.95));rows=[]
 for j,s in enumerate(specs):rows.append(s|{'mean':mu[j],'standard_error':se[j],'descriptive_lower':low[j],'descriptive_upper':high[j],'exploratory_joint_lower':mu[j]-critical*se[j],'exploratory_joint_upper':mu[j]+critical*se[j],'critical_value':critical,'listed_responses':len(specs),'worlds':320,'method':'exploratory centered multinomial world-cluster max-t bootstrap, 5000 replicates; approximate joint bands, not original confirmatory intervals; natural future policy'})
 pd.DataFrame(rows).to_csv(O/'all_one_action_responses_and_bands.csv',index=False);(O/'one_action_band_audit.json').write_text(json.dumps({'listed_responses_and_contrasts':len(specs),'joint_critical_value':critical,'bootstrap_replicates':5000,'worlds':320,'seconds':time.time()-start},indent=2));print('completed response bands',len(specs),'critical',critical,'seconds',round(time.time()-start,1),flush=True)

if __name__ == '__main__':
 main()
