"""Descriptive local recurrence checks; no classification as traps or proven cycles."""
import numpy as np,csv,time,concurrent.futures
from .runtime import WORKERS
from .complete_analysis import prepare,P,OUT,residual
from .oscillation_controls import stats

def process(t):
 split,p,pid,i,eid,path,wp,*_=t
 with np.load(P/wp,allow_pickle=False) as w:cap=w['capacity']
 with np.load(P/path,allow_pickle=False) as f:n=f['stocks'][500:,f['accessible']]/cap[f['accessible']]
 z=residual(n);active=z.var(axis=0)>1e-10;z=z[:,active];r={'evaluation_id':eid,'pressure':p,'index':i,'active_source_fraction':float(active.mean()),'sources_with_resolved_variation':int(active.sum())}
 if z.shape[1]:
  ac=np.column_stack([np.sum(z[:-h]*z[h:],axis=0)/np.sqrt(np.sum(z[:-h]**2,axis=0)*np.sum(z[h:]**2,axis=0)) for h in range(1,21)])
  # a screening shape only: not a significance test, not proof of recurrence.
  r['fraction_with_negative_then_positive_acf_screen']=float(((ac[:,1:5].min(axis=1)<-.1)&(ac[:,5:10].max(axis=1)>.1)).mean());r['mean_source_acf_lag3']=float(ac[:,2].mean());r['mean_source_acf_lag7']=float(ac[:,6].mean());rho,g,period=stats(z.T);r['fraction_whitened_peaks_at5to9']=float(((period>=5)&(period<=9)).mean());r['median_source_whitened_peak_period']=float(np.median(period))
 else:
  for c in ['fraction_with_negative_then_positive_acf_screen','mean_source_acf_lag3','mean_source_acf_lag7','fraction_whitened_peaks_at5to9','median_source_whitened_peak_period']:r[c]=None
 return r

def main():
 tasks=[t for t in prepare() if t[0]=='test' and t[2]=='db0eed394e20c822b2a6beb3'];start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex,(OUT/'baseline_local_recurrence.csv').open('w') as f:
  w=None
  for r in ex.map(process,tasks,chunksize=4):
   if w is None:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader()
   w.writerow(r)
 print('local recurrence checks complete',len(tasks),'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
