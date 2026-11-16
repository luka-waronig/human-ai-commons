"""Exact one-action accounting contrasts at saved states; no future rollout or simulation."""
import pathlib,csv,json,concurrent.futures,time
from functools import lru_cache
import numpy as np
from .runtime import WORKERS
from .complete_analysis import prepare,load_world,P,OUT

def process(t):
 split,p,pid,i,eid,path,wp,drivers,policy=t;w,spec=load_world(wp)
 with np.load(P/path,allow_pickle=False) as f:
  stock=f['stocks'];withdraw=f['source_withdrawal'];target=f['agent_targets'];req=f['agent_requested'];quota=f['agent_reference_request']
 T,N=req.shape;M=stock.shape[1];valid=target>=0;safe=np.maximum(target,0);steps=np.arange(T)[:,None];idx=(steps*M+safe).reshape(-1);total=np.bincount(idx,weights=(req*valid).reshape(-1),minlength=T*M).reshape(T,M);pre=(stock+withdraw)[steps,safe];R=total[steps,safe];kinds=w['kinds'];RC=R-req+quota;RD=R-req+1.5*quota
 fC=np.minimum(1,pre/RC);fD=np.minimum(1,pre/RD);source=(np.maximum(pre-RC,0)-np.maximum(pre-RD,0))/w['capacity'][safe];own=(quota*fC-1.5*quota*fD)/(1.5*quota)
 bygroup={}
 for g,mask in [('human',~kinds),('ai',kinds)]:
  agg=np.bincount(idx[:,None] if False else (steps*M+safe).reshape(-1),weights=(req*mask[None,:]*valid).reshape(-1),minlength=T*M).reshape(T,M)
  other=agg[steps,safe]-req*mask[None,:]
  others=other*(fC-fD)/(1.5*quota[mask][0]);bygroup[g]=(others+own*mask[None,:])/mask.sum()
 r={'evaluation_id':eid,'split':split,'pressure':p,'policy_id':pid,'index':i}
 for phase,start,end in [('formation',0,450),('late',500,T-50)]:
  for donor,mask in [('human',~kinds),('ai',kinds)]:
   m=valid[start:end]&mask[None,:];r[phase+'_exact_'+donor+'_action_source_stock_h0']=float(source[start:end][m].mean());r[phase+'_exact_'+donor+'_own_support_h0']=float(own[start:end][m].mean())
   for recipient,y in bygroup.items():r[phase+'_exact_'+donor+'_action_to_'+recipient+'_support_h0']=float(y[start:end][m].mean())
  r[phase+'_exact_action_source_stock_h0']=float(source[start:end][valid[start:end]].mean())
 return r

def main():
 tasks=prepare();start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex, (OUT/'exact_immediate_effects.csv').open('w') as f:
  writer=None
  for j,r in enumerate(ex.map(process,tasks,chunksize=4),1):
   if writer is None:writer=csv.DictWriter(f,fieldnames=list(r));writer.writeheader()
   writer.writerow(r)
   if j%2000==0:print('exact accounting',j,'/',len(tasks),'seconds',round(time.time()-start,1),flush=True)
 print('completed exact accounting',len(tasks),'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
