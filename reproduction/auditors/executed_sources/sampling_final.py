from pathlib import Path
import sys,copy,json,math,itertools,hashlib
import numpy as np,torch
ROOT=Path('/home/njit/tas_vp_final_20261029');sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp3 import exp3_governance_tasv2 as gov
from sampled_weight_audit import build_admissibility_bounds,verify_sampled_bounds
torch.set_num_threads(2)
base=Path('/home/njit/round2_final_20261028/results/c2/captured');records=[];selected=[]
# Find reproducible sampled-client seeds for every subset of each size.
seeds={}
for k in range(1,11):
 found={};s=0;required=math.comb(10,k)
 while len(found)<required:
  subset=tuple(sorted(np.random.RandomState(s).choice(10,size=k,replace=False).tolist()))
  found.setdefault(subset,s);s+=1
  assert s<100000
 seeds[k]=found
for path in sorted(base.glob('round_*.pt')):
 c=torch.load(path,map_location='cpu',weights_only=False);u=c['local_updates'];bounds=build_admissibility_bounds(u,c['prior_trust_state'])
 ema,hist,anc,tmp,_,_=copy.deepcopy(c['prior_trust_state'])
 truth=gov.full_trust({i:gov.flatten_update(v) for i,v in u.items()},trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,refine_anchor=True,ema_weights=ema,ema_alpha=.7,cos_history=hist,prev_anchor=anc,temporal_anchor=tmp)[0]
 positive=[i for i in truth if bounds['intervals'][i][0]>1e-4 and truth[i]>1e-4]
 if len(positive)<3:continue
 proposals=[]
 for b in [1,2,3]:
  victims=positive[:b];pub=dict(truth);mass=sum(pub[i] for i in victims)
  for i in victims:pub[i]=0.
  others=[i for i in truth if i not in victims]
  for i in others:
   room=max(0.,bounds['intervals'][i][1]-pub[i]-1e-5);delta=min(mass,room);pub[i]+=delta;mass-=delta
  if mass>1e-9:break
  assert abs(sum(pub.values())-1.)<1e-8
  violated=[i for i in range(10) if pub.get(i,0)<bounds['intervals'].get(i,(0,0))[0]-1e-6 or pub.get(i,0)>bounds['intervals'].get(i,(0,0))[1]+1e-6]
  assert sorted(violated)==sorted(victims)
  proposals.append((b,pub,violated))
 if len(proposals)!=3:continue
 selected.append(dict(round=c['round'],capture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),honest_weights=truth,bounds=bounds['intervals'],proposals=[dict(b=b,weights=p,violated=v) for b,p,v in proposals]))
 for b,pub,violated in proposals:
  assert gov.v1_verify_trust(pub,truth,10)[0]
  for k in range(1,11):
   tp=0;fp=0
   for subset,seed in seeds[k].items():
    bad,sampled=verify_sampled_bounds(pub,bounds,10,seed,sample_ratio=k/10)
    honest_bad,_=verify_sampled_bounds(truth,bounds,10,seed,sample_ratio=k/10)
    assert set(sampled)==set(subset)
    assert bool(bad)==bool(set(subset)&set(violated));tp+=bool(bad);fp+=bool(honest_bad)
   n=math.comb(10,k);expected=1-(math.comb(10-b,k) if k<=10-b else 0)/n
   assert fp==0 and abs(tp/n-expected)<1e-12
   records.append(dict(round=c['round'],b=b,k=k,subsets=n,detected=tp,honest_rejected=fp,rate=tp/n))
 if len(selected)==3:break
assert len(selected)==3
out=Path('/home/njit/tas_vp_final_analysis_20261029/sampling_final.json')
result=dict(scope='Conditional coverage for b per-client admissibility violations; sum and nonnegativity remain valid. Not the probability of detecting arbitrary altered coefficients.',selected=selected,records=records,total_subset_proposal_cases=sum(x['subsets'] for x in records),total_honest_false_rejections=sum(x['honest_rejected'] for x in records),source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'sampled_weight_audit.py',ROOT/'scripts/exp3/exp3_governance_tasv2.py']})
out.write_text(json.dumps(result,indent=2));print(json.dumps(dict(rounds=[c['round'] for c in selected],cases=result['total_subset_proposal_cases'],false_rejections=0)))
