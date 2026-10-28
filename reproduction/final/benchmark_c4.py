"""Fresh local complete-verifier controls with matched two-endorsement comparison."""
from pathlib import Path
import copy,hashlib,io,json,secrets,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp3 import exp3_governance_tasv2 as m
OUT=ROOT/'results/c4';OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
def weights(inputs,state):
 ema,history,anchor,temporal,_,_=copy.deepcopy(state)
 return m.full_trust({c:m.flatten_update(u) for c,u in inputs.items()},trust_power=5.,max_weight_ratio=1.5,
   norm_penalty_strength=.8,refine_anchor=True,ema_weights=ema,ema_alpha=.7,
   cos_history=history,prev_anchor=anchor,temporal_anchor=temporal)[0]
def check(role,proposal,capture,inputs):
 reference=weights(inputs,capture['prior_trust_state'])
 binding=m.proposal_transcript_matches(proposal['hash'],round_number=capture['round'],parent_hash=capture['parent_hash'],
   aggregate=proposal['aggregate'],published_weights=proposal['weights'],
   update_hashes={c:m.hash_local_update(u) for c,u in inputs.items()})
 if role=='complete':
  badw=any(abs(proposal['weights'].get(c,0.)-reference.get(c,0.))>1e-6 for c in range(10))
  bada=m.v2_verify_agg(proposal['aggregate'],proposal['weights'],inputs)
 elif role=='vs':
  badw,_,_=m.v1_verify_trust(proposal['weights'],reference,10)
  challenge=m.make_postcommit_challenge(m.flatten_update(proposal['aggregate']).size,secrets.randbits(128))
  bada,_,_=m.v1_verify_agg(proposal['aggregate'],proposal['weights'],inputs,challenge=challenge)
 else:
  badw,_=m.v2_verify_trust(proposal['weights'],reference,10,sample_ratio=.3,seed=42+capture['round']*137)
  bada=m.v2_verify_agg(proposal['aggregate'],proposal['weights'],inputs)
 return not (badw or bada or not binding)
def proposal(capture,inputs,published,aggregate,round_override=None,parent_override=None):
 h=m.proposal_commitment(round_number=capture['round'] if round_override is None else round_override,
    parent_hash=capture['parent_hash'] if parent_override is None else parent_override,
    aggregate=aggregate,published_weights=published,update_hashes={c:m.hash_local_update(u) for c,u in inputs.items()})
 return dict(weights=published,aggregate=aggregate,hash=h)
def altered(aggregate):
 v=m.flatten_update(aggregate);rng=np.random.default_rng(20261028);u=rng.normal(size=len(v));u/=np.linalg.norm(u)
 v=v+u*(.1*max(np.linalg.norm(v),1e-12));result={};offset=0
 for key,value in aggregate.items():
  n=value.numel();result[key]=torch.from_numpy(v[offset:offset+n].astype(np.float32)).reshape(value.shape);offset+=n
 return result
configs={'One complete verifier':['complete'],'Two complete verifiers':['complete','complete'],'TAS roles':['vs','vp']}
observations=[];coverage=[];payloads=[]
for round_number in [1,3,10,25,50]:
 cap=torch.load(ROOT/'results/c2/captured'/f'round_{round_number:02d}.pt',map_location='cpu',weights_only=False)
 inputs=cap['local_updates'];reference=weights(inputs,cap['prior_trust_state']);agg=m.weighted_average_updates(inputs,reference)
 fake=dict(reference);source=max(fake,key=fake.get);target=min((c for c in fake if c!=source),key=fake.get)
 shift=min(.05,fake[source]/2);fake[source]-=shift;fake[target]+=shift
 cases={
  'honest':proposal(cap,inputs,reference,agg),
  'weight_tampering':proposal(cap,inputs,fake,m.weighted_average_updates(inputs,fake)),
  'aggregate_tampering':proposal(cap,inputs,reference,altered(agg)),
  'joint_tampering':proposal(cap,inputs,fake,altered(agg)),
  'stale_round':proposal(cap,inputs,reference,agg,round_override=round_number-1),
  'stale_parent':proposal(cap,inputs,reference,agg,parent_override='0'*64)}
 payload=io.BytesIO();torch.save(dict(round=round_number,parent_hash=cap['parent_hash'],
   inputs=inputs,trust_parent=cap['prior_trust_state'],proposal=cases['honest']),payload)
 payloads.append(dict(round=round_number,clients=len(inputs),serialized_input_bytes=len(payload.getvalue())))
 for case,p in cases.items():
  for name,roles in configs.items():
   # Warm execution is discarded; all measured checks recompute their bound inputs afresh.
   for role in roles:check(role,p,cap,inputs)
   times=[];recoveries=[];accept=None
   for repeat in range(11):
    start=time.perf_counter_ns();flags=[check(role,p,cap,inputs) for role in roles];accept=all(flags);times.append((time.perf_counter_ns()-start)/1e6)
    if case!='honest':
     start=time.perf_counter_ns();w=weights(inputs,cap['prior_trust_state']);candidate=m.weighted_average_updates(inputs,w)
     recovery=proposal(cap,inputs,w,candidate)
     assert all(check(role,recovery,cap,inputs) for role in roles)
     assert m.aggregate_relative_error(candidate,agg)==0.
     recoveries.append((time.perf_counter_ns()-start)/1e6)
   assert accept==(case=='honest'),(name,round_number,case)
   coverage.append(dict(configuration=name,round=round_number,case=case,accepted=accept))
   observations.append(dict(configuration=name,round=round_number,case=case,verify_ms=times,recovery_ms=recoveries,
     logical_received_bytes=len(payload.getvalue())*len(roles)))
 print('Verifier controls completed captured round',round_number,flush=True)
summary={}
for name in configs:
 rows=[r for r in observations if r['configuration']==name]
 verification=np.array([v for r in rows for v in r['verify_ms']]);recovery=np.array([v for r in rows for v in r['recovery_ms']])
 checks=[c for c in coverage if c['configuration']==name]
 summary[name]=dict(honest_accepted=sum(c['accepted'] for c in checks if c['case']=='honest'),honest_cases=5,
    attacks_rejected=sum(not c['accepted'] for c in checks if c['case']!='honest'),attack_cases=25,
    verify_median_ms=float(np.median(verification)),verify_q1_q3=np.percentile(verification,[25,75]).tolist(),
    recovery_median_ms=float(np.median(recovery)),recovery_q1_q3=np.percentile(recovery,[25,75]).tolist(),
    logical_received_MiB_median=float(np.median([r['logical_received_bytes']/2**20 for r in rows])))
save(OUT/'observations.json',observations);save(OUT/'coverage.json',coverage)
save(OUT/'summary.json',dict(comment='R4 round2 Comment 4',configurations=summary,input_payloads=payloads,
  scope='Single-host fresh checks on five captured rounds, 11 timing repetitions per case. Logical serialized receive volume, not measured network traffic. Recovery includes candidate recomputation and the same endorsement requirements as verification.',
  inputs='Canonical bound update/weight tensors with honest, weight/aggregate/joint tampering and stale round/parent cases; not an arbitrary-input fuzzer.',
  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'scripts/exp2/exp2_run.py',ROOT/'scripts/exp3/exp3_governance_tasv2.py']}))
print(json.dumps(summary),flush=True)
