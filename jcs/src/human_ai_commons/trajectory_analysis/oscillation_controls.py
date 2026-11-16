"""Statistical AR(1) surrogates for baseline cooperation spectra; no commons model runs."""
import numpy as np,csv,concurrent.futures,time
from scipy import signal
from .runtime import WORKERS
from .complete_analysis import prepare,P,OUT
B=499

def stats(y):
 y=np.asarray(y,float);t=np.arange(y.shape[-1],dtype=float);t-=t.mean();z=y-y.mean(axis=-1,keepdims=True);z=z-(z@t)[...,None]*t/(t@t);rho=np.sum(z[...,:-1]*z[...,1:],axis=-1)/np.sum(z[...,:-1]**2,axis=-1);rho=np.clip(rho,-.95,.95);e=z[...,1:]-rho[...,None]*z[...,:-1];power=abs(np.fft.rfft(e,axis=-1))**2;f=np.fft.rfftfreq(e.shape[-1]);valid=(f>0)&(f<.5);band=valid&(f>=1/50)&(f<=1/3);g=np.max(power[...,band],axis=-1)/power[...,valid].sum(axis=-1);peak=np.flatnonzero(band)[np.argmax(power[...,band],axis=-1)];return rho,g,1/f[peak]
def process(task):
 split,p,pid,i,eid,path,*_=task
 with np.load(P/path,allow_pickle=False) as z:y=z['strategies'][500:].mean(axis=1)
 rho,g,period=stats(y);gen=np.random.default_rng(20261113+int(p*10000)+i);sur=signal.lfilter([1],[1,-float(rho)],gen.normal(size=(B,len(y)+256)),axis=-1)[:,256:];_,gs,_=stats(sur)
 return {'evaluation_id':eid,'pressure':p,'index':i,'fitted_ar1':float(rho),'peak_period':float(period),'peak_statistic':float(g),'ar1_surrogate_p':float((1+(gs>=g).sum())/(B+1)),'surrogate_series':B}
def main():
 tasks=[t for t in prepare() if t[0]=='test' and t[2]=='db0eed394e20c822b2a6beb3'];start=time.time()
 with concurrent.futures.ProcessPoolExecutor(max_workers=WORKERS) as ex, (OUT/'baseline_oscillation_controls.csv').open('w') as f:
  w=None
  for j,r in enumerate(ex.map(process,tasks,chunksize=2),1):
   if w is None:w=csv.DictWriter(f,fieldnames=list(r));w.writeheader()
   w.writerow(r)
   if j%160==0:print('baseline spectral controls',j,'/',len(tasks),'seconds',round(time.time()-start,1),flush=True)
 print('completed baseline spectral controls',len(tasks),'seconds',round(time.time()-start,1),flush=True)
if __name__=='__main__':main()
