import pathlib,json,hashlib,csv,collections,numpy as np,pandas as pd
from .runtime import P, OUT, O, WORKERS
def main():
 # independent small allocation calculation, compared with closed-form one-action contrasts.
 d=json.loads((P/'test_comparisons.json').read_text());g=next(g for g in d['problems'] if g['problem']==d['primary_problem']);eid=g['families']['reference'][0]['realization_ids'][0];r=next(r for r in csv.DictReader((P/'analysis/realization_metrics.csv').open()) if r['evaluation_id']==eid)
 with np.load(P/r['trajectory_path'],allow_pickle=False) as f:stocks=f['stocks'];withdraw=f['source_withdrawal'];targets=f['agent_targets'];req=f['agent_requested'];q=f['agent_reference_request'];kinds=f['agent_is_ai']
 with np.load(P/r['world_path'],allow_pickle=False) as f:cap=f['capacity']
 err=0.;gen=np.random.default_rng(20261113)
 for t,i in zip(gen.integers(0,len(stocks),size=80),gen.integers(0,len(q),size=80)):
  pre=stocks[t]+withdraw[t];j=targets[t,i];R=req[t,targets[t]==j].sum();rc=R-req[t,i]+q[i];rd=R-req[t,i]+1.5*q[i];fc=min(1,pre[j]/rc);fd=min(1,pre[j]/rd);ds=(max(pre[j]-rc,0)-max(pre[j]-rd,0))/cap[j];own=(q[i]*fc-1.5*q[i]*fd)/(1.5*q[i]);direct=[]
  for choice in [1,1.5]:
   requests=req[t].copy();requests[i]=choice*q[i];totals=np.bincount(targets[t],weights=requests,minlength=len(cap));ratio=np.minimum(1,np.divide(pre,totals,out=np.ones_like(pre),where=totals>0));delivery=requests*ratio[targets[t]];post=np.maximum(0,pre-totals);direct.append((post,delivery/(1.5*q)))
  err=max(err,abs((direct[0][0][j]-direct[1][0][j])/cap[j]-ds),abs(direct[0][1][i]-direct[1][1][i]-own))
  for mask in [~kinds,kinds]:
   other=req[t,(targets[t]==j)&mask].sum()-(req[t,i] if mask[i] else 0);formula=(other*(fc-fd)/(1.5*q[mask][0])+own*mask[i])/mask.sum();err=max(err,abs((direct[0][1][mask]-direct[1][1][mask]).mean()-formula))
 assert err<1e-12,err
 # input/source verification and split independence.
 man=json.loads((P/'manifest.json').read_text());am=json.loads((P/'analysis/analysis_manifest.json').read_text());source_checks={}
 for f,h in man['source_sha256'].items():source_checks[f]=hashlib.sha256((P/'source_snapshot'/f).read_bytes()).hexdigest()==h
 input_checks={}
 for f,h in am['input_sha256'].items():
  sha=hashlib.sha256()
  with (P/f).open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024*1024),b''):sha.update(chunk)
  input_checks[f]=sha.hexdigest()==h
 assert all(source_checks.values()) and all(input_checks.values())
 sets=collections.defaultdict(set)
 for x in csv.DictReader((P/'analysis/realization_metrics.csv').open()):sets[x['split']].add((x['world_id'],x['driver_seed']))
 checks={}
 for a,b in [('train','validation'),('train','test'),('validation','test'),('characterization','test'),('test','robustness')]:checks[a+'__'+b]=len(sets[a]&sets[b]);assert not checks[a+'__'+b]
 (O/'provenance_and_verification.json').write_text(json.dumps({'run_id':man['run_id'],'configuration_id':man['configuration_id'],'archived_source_hash_checks':source_checks,'original_analysis_input_hash_checks':input_checks,'independent_allocation_check_max_error':err,'cross_split_world_driver_pair_overlaps':checks,'scope':'Verification of archived model sources, inputs, independent world-driver splits and one-action allocation accounting.'},indent=2))
 print('verification passed','allocation max error',err,'input hashes',len(input_checks),'source hashes',len(source_checks),flush=True)

if __name__ == '__main__':
 main()
