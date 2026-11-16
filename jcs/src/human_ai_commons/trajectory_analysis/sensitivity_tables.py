import pandas as pd,numpy as np,pathlib,json
from scipy.stats import beta
from .runtime import P, OUT, O, WORKERS
def main():
 s=pd.read_csv(O/'recovered_static_metrics.csv',usecols=['evaluation_id','split','pressure','policy_id']);s=s[s.split=='test'];keys=['pressure','policy_id']
 # the original stock condition is retained as a secondary outcome, never labelled a trap diagnostic.
 t=pd.read_csv(P/'analysis/stress_definitions.csv');t=t.drop(columns='policy_id').merge(s,on='evaluation_id',validate='many_to_one');g=t.groupby(keys+['reference','duration_steps']).agg(realizations=('persistent_local_stress','size'),events=('persistent_local_stress','sum'),mean_affected_fraction=('persistent_cell_fraction','mean')).reset_index();K=len(g);g['event_probability']=g.events/g.realizations;g['simultaneous_probability_upper']=np.where(g.events==g.realizations,1,beta.ppf(1-.05/K,g.events+1,g.realizations-g.events));g['simultaneous_probability_lower']=np.where(g.events==0,0,beta.ppf(.05/K,g.events,g.realizations-g.events+1));g['scope']='secondary stock definition; same trajectories, not a trap definition or reoptimization';g.to_csv(O/'stock_definition_sensitivity.csv',index=False)
 w=pd.read_csv(P/'analysis/welfare_assumptions.csv');w=w.drop(columns='policy_id').merge(s,on='evaluation_id',validate='many_to_one');wm=w.groupby(keys+['utility','additional_use_value','human_weight']).agg(realizations=('welfare','size'),human=('human_service','mean'),ai=('ai_service','mean'),welfare=('welfare','mean')).reset_index();wm.to_csv(O/'welfare_sensitivity_means.csv',index=False)
 rows=[]
 for f in (P/'analysis').glob('depletion-floor-*.csv'):
  d=pd.read_csv(f,usecols=['evaluation_id','loss','floor','tail']);d=d.merge(s,on='evaluation_id',validate='one_to_one');z=d.groupby(keys+['floor','tail']).agg(realizations=('loss','size'),depletion=('loss','mean')).reset_index();rows.append(z)
 pd.concat(rows).to_csv(O/'depletion_definition_sensitivity.csv',index=False)
 c=pd.read_json(P/'capacity_bounds.jsonl',lines=True);c.groupby('pressure').agg(worlds=('world_id','size'),strict_zero_depletion_obstruction=('strict_safety_ruled_out','sum')).reset_index().to_csv(O/'capacity_scope_summary.csv',index=False)
 print('completed sensitivity tables','stress conditions',len(g),'welfare conditions',len(wm),'min stress event rate',g.event_probability.min(),flush=True)

if __name__ == '__main__':
 main()
