"""Recovered spatial organization and service distributions from all simulation summaries."""
import pathlib,json,csv,time,collections
from functools import lru_cache
import numpy as np
from .runtime import P, OUT, O, WORKERS
def correlation(a,b):
 a=np.asarray(a,float)-np.mean(a);b=np.asarray(b,float)-np.mean(b);den=np.linalg.norm(a)*np.linalg.norm(b);return float(a@b/den) if den>1e-14 else None
def moran(a,W):
 a=np.asarray(a,float)-np.mean(a);den=a@a;return float(len(a)/W.sum()*(a@W@a)/den) if den>1e-16 and W.sum()>0 else None
def tail(v,f=.1):
 v=np.sort(v);q=len(v)*f;k=int(q);return float((v[:k].sum()+(q-k)*v[k])/q) if k<len(v) else float(v.mean())
def gini(v):
 v=np.sort(v);n=len(v);return float(2*np.arange(1,n+1)@v/(n*v.sum())-(n+1)/n) if v.sum()>0 else 0.
@lru_cache(maxsize=192)
def geometry(path):
 with np.load(P/path,allow_pickle=False) as f:positions=f['positions'];resources=f['resources'];kinds=f['kinds'];renewal=f['renewal'];spec=json.loads(str(f['spec']))
 dims=np.array([spec['width'],spec['height']]);d=abs(resources[:,None]-positions[None,:]);d=np.minimum(d,dims-d).max(axis=-1);accessible=(d<=1).any(axis=1);eco=(d<=3).astype(float);E=eco/np.maximum(eco.sum(axis=1)[:,None],1)
 dp=abs(positions[:,None]-positions[None,:]);dp=np.minimum(dp,dims-dp).max(axis=-1);W=(dp<=1).astype(float);np.fill_diagonal(W,0)
 dr=abs(resources[accessible,None]-resources[None,accessible]);dr=np.minimum(dr,dims-dr).max(axis=-1);Wr=((dr<=1)&(dr>0)).astype(float)
 return kinds,E[accessible],W,Wr,accessible,float(renewal[accessible].sum())
def main():
 start=time.time();stats=collections.Counter();maxerr=0;fields=None
 with (O/'recovered_static_metrics.csv').open('w') as out:
  writer=None
  for line in (P/'realizations.jsonl').open():
   z=json.loads(line);kinds,E,W,Wr,accessible,supply=geometry(z['world_path']);base=np.array(z['agent_reference_request']);req=np.array(z['agent_mean_requested']);C=2*(1.5-req/base);service=np.array(z['agent_mean_service']);depleted=np.array(z['cell_depleted_fraction']);duration=np.array(z['site_longest_continuous_stress_steps']);err=abs(C.mean()-z['mean_cooperation']);maxerr=max(maxerr,err)
   if np.max(C)>1+1e-10 or np.min(C)<-1e-10 or err>1e-10:raise ValueError(('cooperation recovery',z['evaluation_id'],err))
   r={k:z[k] for k in ['evaluation_id','split','policy_id','variant','mean_cooperation','human_service','ai_service','loss','mean_accessible_resource','mean_actual_renewal','mean_delivered','mean_unmet','persistent_cell_fraction','persistent_local_stress','observation_steps']}
   r.update(pressure=z['scenario']['pressure'],u=z['policy']['u'],v=z['policy']['v'],observation=z['policy']['observation'],index=z['index'],world_id=z['world_id'],baseline_supply=supply,historical_default_equivalent_pressure=105/supply,cooperation_reconstruction_error=err,cooperation_moran=moran(C,W),depleted_time_fraction_moran=moran(depleted,Wr),local_cooperation_depleted_time_association=correlation(E@C,depleted),mean_continuous_stress_duration_fraction=float(np.mean(duration/z['observation_steps'])),full_window_stressed_resource_fraction=float(np.mean(duration==z['observation_steps'])),service_gini_all=gini(service))
   residual=service.copy()
   for name,mask in [('human',~kinds),('ai',kinds)]:
    v=service[mask];r[name+'_minimum_agent_support']=float(v.min());r[name+'_worst10_agent_support']=tail(v);r[name+'_support_gini']=gini(v);r[name+'_cooperation_moran']=moran(C[mask],W[np.ix_(mask,mask)]);r[name+'_mean_cooperation']=float(C[mask].mean());residual[mask]-=v.mean()
   r['service_moran_after_type_centering']=moran(residual,W)
   if writer is None:writer=csv.DictWriter(out,fieldnames=list(r));writer.writeheader()
   writer.writerow(r);stats[z['split']]+=1
   if sum(stats.values())%10000==0:print('recovered',sum(stats.values()),'seconds',round(time.time()-start,1),flush=True)
 (O/'static_audit.json').write_text(json.dumps({'rows':dict(stats),'maximum_cooperation_reconstruction_error':maxerr,'elapsed_seconds':time.time()-start},indent=2));print('completed',dict(stats),'max error',maxerr,'seconds',round(time.time()-start,1),flush=True)

if __name__ == '__main__':
 main()
