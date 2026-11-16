"""Exploratory analysis of archived trajectories; no simulator imports or model execution."""
import os,json,csv,pathlib,collections,math,sys,concurrent.futures,time
import numpy as np
from functools import lru_cache
@lru_cache(maxsize=1)
def baseline_events():
 path=OUT/"baseline_events.json"
 return json.loads(path.read_text()) if path.exists() else {}
from .runtime import P, OUT, O, WORKERS
LAGS=[-50,-20,-10,-5,-3,-2,-1,0,1,2,3,5,10,20,50]
BINS=np.array([0,.1,.25,.4,.5,.6,.75,.9,1.0000001])
def corr(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float);a=a-a.mean(axis=0);b=b-b.mean(axis=0)
 den=np.sqrt(np.sum(a*a)*np.sum(b*b))
 return float(np.sum(a*b)/den) if den>1e-16 else None
def spatialcorr(a,b):
 return corr(a[:,None],b[:,None])
def lagcurve(n,x):
 out=[]
 for k in LAGS:
  if k>0:a,b=n[:-k],x[k:]
  elif k<0:a,b=n[-k:],x[:k]
  else:a,b=n,x
  out.append(corr(a,b))
 return out
def process_loaded(task,w,spec,data):
 pressure,policy_id,index,eid,trajectory,worldpath=task
 stocks=data['stocks'];S=data['strategies'].astype(float);targets=data['agent_targets'];withdraw=data['source_withdrawal'];renew=data['resource_renewal'];requests=data['agent_requested'];delivered=data['agent_delivered'];accessible=data['accessible'];base=data['agent_reference_request']
 cap=w['capacity'];n=stocks/cap;pre=(stocks+withdraw)/cap;T,N=S.shape;M=len(cap)
 delta=abs(w['resources'][:,None,:]-w['positions'][None,:,:]);delta=np.minimum(delta,np.array([spec['width'],spec['height']])-delta);dist=delta.max(axis=-1)
 ecology=(dist<=3).astype(float);count=ecology.sum(axis=1);incidence=(ecology/np.maximum(count[:,None],1)).T
 X=S@incidence
 ts=np.arange(T)[:,None];valid=targets>=0;seen=pre[ts,np.maximum(targets,0)];endseen=n[ts,np.maximum(targets,0)]
 r={'pressure':pressure,'policy_id':policy_id,'index':index,'evaluation_id':eid}
 profile={}
 r['feedback_neighbors_with_direct_access_fraction']=float(((dist<=1).sum(axis=1)/np.maximum(count,1))[accessible].mean())
 r['target_change_fraction']=float((targets[1:]!=targets[:-1])[valid[1:]&valid[:-1]].mean())
 at_risk=(endseen[:-1]<.5)&valid[:-1]&valid[1:]
 hidden=at_risk&(seen[1:]>.5)&(targets[1:]!=targets[:-1])
 r['next_abundant_other_source_given_depleted_last_source']=float(hidden.sum()/at_risk.sum()) if at_risk.any() else None
 r['depleted_to_abundant_other_source_fraction_all_steps']=float(hidden.sum()/at_risk.size)
 profile['mean_stock']=n[:,accessible].mean(axis=1).tolist();profile['mean_cooperation']=S.mean(axis=1).tolist()
 for name,start,end in [('first100',0,100),('formation',0,500),('late',500,T)]:
  nwin=n[start:end,accessible];swin=S[start:end];xwin=X[start:end,accessible]
  r[name+'_mean_stock']=float(nwin.mean());r[name+'_mean_cooperation']=float(swin.mean())
  r[name+'_stock_temporal_sd']=float(nwin.mean(axis=1).std());r[name+'_cooperation_temporal_sd']=float(swin.mean(axis=1).std())
  r[name+'_stock_between_location_variance_share']=float(nwin.mean(axis=0).var()/nwin.var()) if nwin.var()>1e-20 else None
  r[name+'_cooperation_between_agent_variance_share']=float(swin.mean(axis=0).var()/swin.var()) if swin.var()>1e-20 else None
  support=np.minimum(delivered[start:end]/(1.5*base),1)
  r[name+'_human_support']=float(support[:,~w['kinds']].mean());r[name+'_ai_support']=float(support[:,w['kinds']].mean())
  if name in ['formation','late']:
   profile[name+'_stock_to_neighborhood_cooperation']=lagcurve(nwin,xwin)
   profile[name+'_dstock_to_dcooperation']=lagcurve(np.diff(nwin,axis=0),np.diff(xwin,axis=0))
  if name=='late':
   for b in range(len(BINS)-1):
    mask=(seen[start:end]>=BINS[b])&(seen[start:end]<BINS[b+1])&valid[start:end]
    r['abundance_bin_'+str(b)+'_cooperation']=float(swin[mask].mean()) if mask.any() else None
    r['abundance_bin_'+str(b)+'_exposure']=float(mask.mean())
 r['late_stock_pattern_first_last_correlation']=spatialcorr(n[500:1000,accessible].mean(axis=0),n[1500:,accessible].mean(axis=0))
 r['late_strategy_pattern_first_last_correlation']=spatialcorr(S[500:1000].mean(axis=0),S[1500:].mean(axis=0))
 # first downcrossing in the first 500 steps; all have at least 1,500 observed future steps.
 # these thresholds locate episodes for analysis; none defines a trap or acceptable design.
 site_events={}
 for threshold in [.25,.5,.75]:
  prev=np.concatenate([np.ones((1,M)),n[:-1]],axis=0)
  crossing=(prev[:500]>=threshold)&(n[:500]<threshold)&accessible[None,:]
  has=crossing.any(axis=0);sites=np.flatnonzero(has);first=crossing[:,sites].argmax(axis=0) if len(sites) else np.array([],int)
  site_events[str(threshold)]=np.where(has,crossing.argmax(axis=0),-1).tolist()
  prefix='firstcross_'+str(threshold)+'_'
  r[prefix+'fraction_locations']=float(has[accessible].mean())
  r[prefix+'mean_step']=float(first.mean()) if len(sites) else None
  if len(sites):
   future=n[first[None,:]+np.arange(1,501)[:,None],sites[None,:]]>=threshold
   returned=future.any(axis=0);duration=np.where(returned,future.argmax(axis=0)+1,500)
   r[prefix+'restricted_mean_return_steps_500']=float(duration.mean())
   for h in [5,20,50,100,500]:r[prefix+'return_by_'+str(h)]=float(future[:h].any(axis=0).mean())
  else:
   r[prefix+'restricted_mean_return_steps_500']=None
   for h in [5,20,50,100,500]:r[prefix+'return_by_'+str(h)]=None
  # event profiles for the half-capacity decision boundary, requiring prior observations.
  if threshold==.5:
   keep=first>=20;st=sites[keep];ft=first[keep];r['firstcross_event_profile_locations']=len(st)
   if len(st):
    times=ft[None,:]+np.arange(-20,101)[:,None]
    profile['firstcross_stock']=n[times,st[None,:]].mean(axis=1).tolist()
    profile['firstcross_neighborhood_cooperation']=X[times,st[None,:]].mean(axis=1).tolist()
    profile['firstcross_renewal']=((renew/cap)[times,st[None,:]]).mean(axis=1).tolist()
    profile['firstcross_withdrawal']=((withdraw/cap)[times,st[None,:]]).mean(axis=1).tolist()
   else:
    for f in ['stock','neighborhood_cooperation','renewal','withdrawal']:profile['firstcross_'+f]=[None]*121
 # fixed baseline event cohort: compare the same sites at the same times under each policy.
 ev=baseline_events().get(str(pressure)+'|'+str(index))
 if ev is not None:
  bf=np.array(ev['0.5']);sites=np.flatnonzero(bf>=0);ft=bf[sites]
  if len(sites):
   for h in [0,20,100,500]:
    r['baseline_event_stock_after_'+str(h)]=float(n[ft+h,sites].mean())
    r['baseline_event_cooperation_after_'+str(h)]=float(X[ft+h,sites].mean())
   prev=np.concatenate([np.ones((1,M)),n[:-1]],axis=0)
   allcross=(prev>=.5)&(n<.5)
   first_full=np.where(allcross.any(axis=0),allcross.argmax(axis=0),T)
   r['baseline_event_no_downcross_by_plus100']=float((first_full[sites]>ft+100).mean())
 return r,profile,site_events
