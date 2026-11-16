"""All stored JCS trajectories, read-only. No commons simulator is imported or executed."""
import pathlib,json,csv,collections,concurrent.futures,time,sys,math,os
from functools import lru_cache
import numpy as np
from scipy import signal
from .basic_calculations import process_loaded,corr
from .runtime import P, OUT, O, WORKERS
HORIZONS=[0,1,2,5,10,20,50]
@lru_cache(maxsize=32)
def load_world(path):
 with np.load(P/path,allow_pickle=False) as f:
  w={k:f[k] for k in ['resources','positions','capacity','kinds','initial','renewal']};spec=json.loads(str(f['spec']))
 return w,spec
@lru_cache(maxsize=12)
def load_reversals(path):
 with np.load(P/path,allow_pickle=False) as f:return f['reversal']
def residual(a):
 a=np.asarray(a,float);t=np.arange(len(a),dtype=float);t-=t.mean();z=a-a.mean(axis=0)
 return z-t.reshape((-1,)+(1,)*(a.ndim-1))*np.sum(t.reshape((-1,)+(1,)*(a.ndim-1))*z,axis=0)/np.sum(t*t)
def spectral(y):
 y=residual(y);den=np.dot(y[:-1],y[:-1]);phi=float(np.clip(np.dot(y[:-1],y[1:])/den,-.95,.95)) if den>1e-20 else 0.
 z=y[1:]-phi*y[:-1];power=abs(np.fft.rfft(z))**2;freq=np.fft.rfftfreq(len(z));valid=(freq>0)&(freq<.5);total=power[valid].sum();band=valid&(freq>=1/50)&(freq<=1/3);around=valid&(freq>=1/9)&(freq<=1/5)
 if total<=1e-20:return {'ar1':phi,'peak_period':None,'whitened_peak_reference_p':None,'power_5to9':None}
 k=np.flatnonzero(band)[np.argmax(power[band])];g=float(power[k]/total);m=int(valid.sum());p=min(1.,float(band.sum())*max(0.,1-g)**(m-1))
 return {'ar1':phi,'peak_period':float(1/freq[k]),'whitened_peak_reference_p':p,'power_5to9':float(power[around].sum()/total)}
def graph_moran(a,W):
 z=np.asarray(a,float)-np.mean(a);den=z@z
 return float(len(z)/W.sum()*(z@W@z)/den) if den>1e-20 and W.sum()>0 else None

def process(task):
 split,pressure,pid,index,eid,traj,worldpath,drivers,policy=task;w,spec=load_world(worldpath)
 with np.load(P/traj,allow_pickle=False) as f:data={k:f[k] for k in ['stocks','strategies','agent_targets','source_withdrawal','resource_renewal','agent_requested','agent_delivered','accessible','agent_reference_request']}
 basic,profiles,events=process_loaded((pressure,pid,index if split=="test" else -1,eid,traj,worldpath),w,spec,data)
 r=basic;r['index']=index;r['split']=split;r['observation']=policy['observation'];r['u']=policy['u'];r['v']=policy['v'];r['forced']=policy.get('forced') or ''
 n=data['stocks']/w['capacity'];S=data['strategies'].astype(float);targets=data['agent_targets'];T,N=S.shape;valid=targets>=0;safe=np.maximum(targets,0);ts=np.arange(T)[:,None];pre=(data['stocks']+data['source_withdrawal'])/w['capacity'];seen=pre[ts,safe];accessible=data['accessible'];kinds=w['kinds'];M=n.shape[1]
 delta=abs(w['resources'][:,None,:]-w['positions'][None,:,:]);delta=np.minimum(delta,np.array([spec['width'],spec['height']])-delta);dist=delta.max(axis=-1);eco=(dist<=3).astype(float);counts=eco.sum(axis=1);E=(eco/np.maximum(counts[:,None],1)).T;X=S@E
 old=np.concatenate([w['initial'][None,:].astype(float),S[:-1]],axis=0)
 if policy.get('forced'):old[0]=policy['forced']=='C'
 local_counts=counts[safe]-1;local_x=np.divide((old@eco.T)[ts,safe]-old,local_counts,out=np.full_like(S,.5),where=local_counts>0)
 global_x=(old.sum(axis=1)[:,None]-old)/(N-1)
 def best(x):
  b=x*(policy['t']-1)+(1-x)*(-policy['s']);c=x*policy['u']+(1-x)*policy['v'];return (2*seen-1)*b-(1-seen)*c<=0
 actual_x=local_x if policy['observation']=='local' else (global_x if policy['observation']=='global' else .5)
 preference=best(actual_x);prob=.1+.8*preference
 if not policy.get('forced'):
  reversal=load_reversals(drivers);recorded_preference=np.logical_xor(S.astype(bool),reversal<.1)
  b=actual_x*(policy['t']-1)+(1-actual_x)*(-policy['s']);c=actual_x*policy['u']+(1-actual_x)*policy['v'];gap=(2*seen-1)*b-(1-seen)*c
  mismatch=(preference!=recorded_preference)&valid;r['reconstructed_action_mismatches']=int(mismatch.sum());r['nonboundary_action_mismatches']=int((mismatch&(abs(gap)>1e-12)).sum())
  if r['nonboundary_action_mismatches']:raise ValueError((eid,'action reconstruction failed',r['nonboundary_action_mismatches']))
  preference=np.where(abs(gap)<1e-12,recorded_preference,preference);prob=.1+.8*preference
 else:r['reconstructed_action_mismatches']=0;r['nonboundary_action_mismatches']=0
 eta=S-prob
 balance=data['stocks']-(np.concatenate([w['capacity'][None,:],data['stocks'][:-1]],axis=0)+data['resource_renewal']-data['source_withdrawal'])
 r['maximum_source_balance_error']=float(abs(balance).max())
 support=np.minimum(data['agent_delivered']/(1.5*data['agent_reference_request']),1);prev_support=np.concatenate([np.full((1,N),2/3),support[:-1]],axis=0)
 for phase,start,end in [('formation',0,500),('late',500,T)]:
  a=n[start:end,accessible];x=X[start:end,accessible];ar=residual(a);xr=residual(x);sg=S[start:end].mean(axis=1);ng=a.mean(axis=1)
  for f,z in [('cooperation',sg),('stock',ng)]:
   for k,val in spectral(z).items():r[phase+'_'+f+'_spectrum_'+k]=val
  freqs,coherence=signal.coherence(residual(sg),residual(ng),fs=1,window='hann',nperseg=128,noverlap=64)
  band=(freqs>=1/9)&(freqs<=1/5);r[phase+'_stock_cooperation_coherence_5to9']=(float(np.mean(coherence[band][np.isfinite(coherence[band])])) if np.isfinite(coherence[band]).any() else None);r[phase+'_stock_cooperation_coherence_max']=(float(np.max(coherence[np.isfinite(coherence)&(freqs>=1/50)&(freqs<=1/3)])) if np.isfinite(coherence).any() else None)
  for lag in [1,2,5,10,20,50,100]:
   r[phase+'_detrended_stock_self_corr_'+str(lag)]=corr(ar[:-lag],ar[lag:]);r[phase+'_detrended_stock_to_coop_'+str(lag)]=corr(ar[:-lag],xr[lag:])
  resource=w['resources'][accessible];dd=abs(resource[:,None,:]-resource[None,:,:]);dd=np.minimum(dd,np.array([spec['width'],spec['height']])-dd);dd=dd.max(axis=-1)
  for radius in [1,3]:
   W=((dd<=radius)&(dd>0)).astype(float);neighbor=ar@(W/np.maximum(W.sum(axis=1)[:,None],1)).T
   for lag in [0,1,2,5,10]:r[phase+'_neighbor_stock_r'+str(radius)+'_lag'+str(lag)]=corr(ar[:-lag] if lag else ar,neighbor[lag:] if lag else neighbor)
  # counterfactual rules at exactly the recorded state; these do not propagate a new trajectory.
  r[phase+'_preference_disagreement_fixed']=float((preference[start:end]!=best(.5)[start:end])[valid[start:end]].mean())
  r[phase+'_preference_disagreement_global']=float((preference[start:end]!=best(global_x)[start:end])[valid[start:end]].mean())
  r[phase+'_expected_cooperation_change_if_fixed']=float((.8*(best(.5)[start:end].astype(float)-preference[start:end]))[valid[start:end]].mean())
  for threshold in [.25,.5,.75]:
   state=a<threshold;r[phase+'_crossings_per_resource_step_'+str(threshold)]=float((state[1:]!=state[:-1]).mean());r[phase+'_downcrossings_per_resource_step_'+str(threshold)]=float((state[1:]&~state[:-1]).mean())
  if not policy.get('forced'):
   t=np.arange(start,end-50);weights=eta[t]/.09;original=safe[t];current_pre=seen[t];prev_own=prev_support[t]
   r[phase+'_randomization_mean_residual']=float(eta[t].mean());r[phase+'_randomization_pre_stock_moment']=float((weights*current_pre).mean())
   kindweights={name:weights[:,mask].mean(axis=1) for name,mask in [('human',~kinds),('ai',kinds)]}
   for h in HORIZONS:
    stock_future=n[(t+h)[:,None],original];own_sensed=seen[t+h]
    for name,y,prevalue in [('source_stock',stock_future,current_pre),('future_sensed_stock',own_sensed,current_pre),('neighborhood_cooperation',X[(t+h)[:,None],original],(old@E)[t[:,None],original]),('source_renewal',data['resource_renewal'][(t+h)[:,None],original]/w['capacity'][original],data['resource_renewal'][t[:,None],original]/w['capacity'][original])]:
     r[phase+'_ipw_'+name+'_h'+str(h)]=float((weights*(y-prevalue))[valid[t]].mean())
    for name,mask in [('human',~kinds),('ai',kinds)]:
     r[phase+'_ipw_own_'+name+'_support_h'+str(h)]=float((weights[:,mask]*(support[t+h][:,mask]-prev_own[:,mask])).mean())
     target_delta=support[t+h][:,mask].mean(axis=1)-prev_own[:,mask].mean(axis=1)
     for donor,dweights in kindweights.items():r[phase+'_ipw_'+donor+'_action_to_'+name+'_support_h'+str(h)]=float((dweights*target_delta).mean())
  # exposure-induced and residual spatial clustering in strategies.
  dp=abs(w['positions'][:,None,:]-w['positions'][None,:,:]);dp=np.minimum(dp,np.array([spec['width'],spec['height']])-dp);W=(dp.max(axis=-1)<=1).astype(float);np.fill_diagonal(W,0)
  r[phase+'_moran_observed_cooperation']=graph_moran(S[start:end].mean(axis=0),W)
  if not policy.get('forced'):
   r[phase+'_moran_expected_cooperation']=graph_moran(prob[start:end].mean(axis=0),W);r[phase+'_moran_unexpected_cooperation']=graph_moran(eta[start:end].mean(axis=0),W)
 # conditional response to own immediately preceding source while holding actual current sensing apart.
 switch=(targets[1:]!=targets[:-1]);end_previous=n[:-1][np.arange(T-1)[:,None],safe[:-1]]
 for label,mask in [('same',~switch),('different',switch)]:
  r['late_next_cooperation_after_low_'+label]=float(S[501:][((end_previous[500:]<.5)&mask[500:]&valid[501:])].mean()) if ((end_previous[500:]<.5)&mask[500:]&valid[501:]).any() else None
 profiles['cooperation_global_series']=S.mean(axis=1).tolist()
 return r,profiles

def prepare():
 d=json.loads((P/'test_comparisons.json').read_text());policies={};ids={}
 for group in d['problems']:
  for cs in group['families'].values():
   for c in cs:
    policies[c['policy_id']]=c['policy']
    for i,e in enumerate(c['realization_ids']):ids[e]=(group['problem']['pressure'],c['policy_id'],i)
 for c in d['same_matrix_observation_controls']:
  for o,v in c['observations'].items():
   policies[v['policy_id']]=v['policy']
   for i,e in enumerate(v['realization_ids']):ids[e]=(1,v['policy_id'],i)
 for g in json.loads((P/'reference_characterization.json').read_text()):
  for c in g['policies'].values():
   policies[c['policy_id']]=c['policy']
   for i,e in enumerate(c['realization_ids']):ids[e]=(g['pressure'],c['policy_id'],i)
 tasks=[]
 for r in csv.DictReader((P/'analysis/realization_metrics.csv').open()):
  if r['trajectory_path']:
   p,pid,i=ids[r['evaluation_id']];tasks.append((r['split'],p,pid,i,r['evaluation_id'],r['trajectory_path'],r['world_path'],r['drivers_path'],policies[pid]))
 tasks.sort(key=lambda t:(t[0],t[3],t[1],t[2]))
 (OUT/'trajectory_inventory.json').write_text(json.dumps({'conditions':dict(collections.Counter((t[0] for t in tasks))),'policies':policies,'run_id':json.loads((P/'manifest.json').read_text())['run_id']},indent=2))
 return tasks

def main():
 tasks=prepare();pilot='--pilot' in sys.argv
 if pilot:tasks=[t for t in tasks if (t[0]=='test' and t[3]<1) or (t[0]=='characterization' and t[3]<1)]
 tag='pilot' if pilot else 'complete';print('saved trajectories',len(tasks),'workers',WORKERS,flush=True);start=time.time();sums={};counts={};rows=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex:
  with (OUT/(tag+'_realizations.jsonl')).open('w') as stream:
   for i,(r,profiles) in enumerate(ex.map(process,tasks,chunksize=2),1):
    stream.write(json.dumps(r,allow_nan=False)+'\n');rows.append(r);key=(r['split'],r['pressure'],r['policy_id'])
    if key not in sums:sums[key]={};counts[key]={}
    for f,v in profiles.items():
     a=np.asarray([np.nan if z is None else z for z in v]);ok=np.isfinite(a)
     if f not in sums[key]:sums[key][f]=np.zeros(len(a));counts[key][f]=np.zeros(len(a),int)
     sums[key][f]+=np.where(ok,a,0);counts[key][f]+=ok
    if i%320==0:print('analyzed',i,'of',len(tasks),'elapsed seconds',round(time.time()-start,1),flush=True)
 with (OUT/(tag+'_realization_metrics.csv')).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
 out=[]
 for key in sums:
  out.append({'split':key[0],'pressure':key[1],'policy_id':key[2],'profiles':{f:np.divide(sums[key][f],counts[key][f],out=np.full(len(v),np.nan),where=counts[key][f]>0).tolist() for f,v in sums[key].items()},'counts':{f:v.tolist() for f,v in counts[key].items()}})
 (OUT/(tag+'_profiles.json')).write_text(json.dumps(out))
 print('analysis completed',len(rows),'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
