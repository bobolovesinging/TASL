"""Stateful raw/CKKS-decoded trust-and-audit replay on a frozen 50-round input sequence."""
from pathlib import Path
import copy,csv,hashlib,json,sys,time
import numpy as np,torch,tenseal as ts
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as m
OUT=ROOT/'results/c7';OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):
 p=path.with_suffix('.tmp');p.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item())+'\n');p.replace(path)
def empty_state():return (None,[],None,None,set(),{c:0 for c in range(10)})
def step(model,state,updates):
 ema,history,anchor,temporal,excluded,violations=copy.deepcopy(state)
 active={c:u for c,u in updates.items() if c not in excluded}
 flat={c:m.flatten_update(u) for c,u in active.items()}
 weights,cosines,new_anchor,u=learning.compute_tasl_final_weights(flat,trust_power=5.,max_weight_ratio=1.5,
    norm_penalty_strength=.8,refine_anchor=True,ema_weights=ema,ema_alpha=.7,
    cos_history=history,prev_anchor=anchor,temporal_anchor=temporal,return_history=True)
 norms={c:float(np.linalg.norm(v)) for c,v in flat.items()}
 median=float(np.median(list(norms.values())))
 vals=list(cosines.values());q1,q3=np.percentile(vals,[25,75]);cutoff=min(.4,max(.2,float(q1-(q3-q1))))
 direction=learning.direction_screening_scores(flat,reference=temporal if temporal is not None else anchor)
 decisions=dict(norm={c:n<=2.5*median for c,n in norms.items()},
                cosine={c:v>=cutoff for c,v in cosines.items()},direction={c:p>0 for c,p in direction.items()})
 if sum(weights.values())<=1e-12:return model,state,weights,cosines,direction,None,decisions
 ema=dict(u);history.append(dict(cosines));anchor=new_anchor
 if len(excluded)<4:
  ids=[c for c in range(10) if c not in excluded]
  if len(ids)>=2:
   mid=float(np.median([np.linalg.norm(flat[c]) for c in ids if c in flat]))+1e-12
   for c in ids:
    if c not in flat:continue
    bad=cosines.get(c,.5)<-.1 or np.linalg.norm(flat[c])/mid>2.
    violations[c]=violations[c]+1 if bad else max(0,violations[c]-1)
    if violations[c]>=2 and len(excluded)<4:excluded.add(c);violations[c]=0
 aggregate=m.weighted_average_updates(active,weights)
 v=m.flatten_update(aggregate)
 if np.linalg.norm(v)>1e-12:temporal=v
 return m.state_add(model,aggregate),(ema,history,anchor,temporal,excluded,violations),weights,cosines,direction,aggregate,decisions

def roundtrip(updates,context):
 decoded={};size=0;enc=dec=0.;maximum=0.
 for c,update in updates.items():
  raw=m.flatten_update(update);chunks=[]
  for start in range(0,len(raw),4096):
   segment=raw[start:start+4096];padding=np.zeros(4096);padding[:len(segment)]=segment
   t=time.perf_counter();blob=ts.ckks_vector(context,padding.tolist()).serialize();enc+=time.perf_counter()-t;size+=len(blob)
   t=time.perf_counter();values=np.asarray(ts.ckks_vector_from(context,blob).decrypt(),dtype=np.float64)[:len(segment)];dec+=time.perf_counter()-t
   maximum=max(maximum,float(np.max(np.abs(values-segment))));chunks.append(values.astype(np.float32))
  v=np.concatenate(chunks);offset=0;decoded[c]={}
  for key,tensor in update.items():
   count=tensor.numel();decoded[c][key]=torch.from_numpy(v[offset:offset+count].copy()).reshape(tensor.shape);offset+=count
 return decoded,dict(ciphertext_bytes=size,encrypt_seconds=enc,decrypt_seconds=dec,max_decode_abs_error=maximum)

source=ROOT/'results/c2';reference=json.loads((source/'reference/round_metrics.json').read_text())
manifest=dict(comment='R4 second round Comment 7',tenseal=ts.__version__,torch=torch.__version__,numpy=np.__version__,
    context=dict(poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60],scale=2**40,threads=2),
    repeats=3,rounds=50,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/exp2/exp2_run.py',ROOT/'scripts/exp3/exp3_governance_tasv2.py']},
    captured_index_sha256=sha(source/'captured_index.json'),
    scope='Repeated cryptographic executions of one frozen external client-update sequence; recomputed histories/exclusions/audits/model accumulation, not client retraining under decoded-model feedback.')
save(OUT/'manifest.json',manifest)
all_rows=[];boundary=[]
for repeat in range(3):
 context=ts.context(ts.SCHEME_TYPE.CKKS,poly_modulus_degree=8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=2)
 context.global_scale=2**40
 context.generate_galois_keys()
 raw_state=empty_state();decoded_state=empty_state();raw_model=None;decoded_model=None
 for r in range(1,51):
  path=source/'captured'/f'round_{r:02d}.pt';capture=torch.load(path,map_location='cpu',weights_only=False)
  inputs=capture['local_updates']
  if r==1:raw_model=copy.deepcopy(capture['parent_model']);decoded_model=copy.deepcopy(raw_model)
  assert raw_state[4]==set(capture['prior_trust_state'][4])
  assert decoded_state[4]==raw_state[4], 'Roster changed; frozen sequence cannot supply missing counterfactual client updates'
  decoded,crypto=roundtrip(inputs,context)
  pre_raw=copy.deepcopy(raw_state);pre_dec=copy.deepcopy(decoded_state)
  raw_model,raw_state,w0,c0,d0,a0,g0=step(raw_model,raw_state,inputs)
  decoded_model,decoded_state,w1,c1,d1,a1,g1=step(decoded_model,decoded_state,decoded)
  raw_hash=m.protocol_state_hash(raw_model,*raw_state)
  assert raw_hash==reference[r-1]['state_hash'], 'Plain replay is not the captured protocol trajectory'
  keys=set(w0)|set(w1)
  row=dict(repeat=repeat,round=r,raw_state_matches_training=True,
    max_weight_abs_error=max(abs(w0.get(c,0.)-w1.get(c,0.)) for c in keys),
    max_cosine_abs_error=max(abs(c0[c]-c1[c]) for c in c0),
    max_direction_abs_error=max(abs(d0[c]-d1[c]) for c in d0),
    norm_decision_changes=sum(g0['norm'][c]!=g1['norm'][c] for c in g0['norm']),
    cosine_decision_changes=sum(g0['cosine'][c]!=g1['cosine'][c] for c in g0['cosine']),
    direction_decision_changes=sum(g0['direction'][c]!=g1['direction'][c] for c in g0['direction']),
    exclusion_sets_match=raw_state[4]==decoded_state[4],
    model_relative_error=m.aggregate_relative_error(decoded_model,raw_model),**crypto)
  if a1 is not None:
   reverse=m.weighted_average_updates(dict(reversed(list(decoded.items()))),w1)
   honest=m.aggregate_relative_error(reverse,a1)
   challenge=m.make_postcommit_challenge(m.flatten_update(a1).size,repeat*1000+r)
   s_bad,_,_=m.v1_verify_trust(w1,w1,10);p_bad,_=m.v2_verify_trust(w1,w1,10,seed=42+r)
   v_bad=m.v2_verify_agg(reverse,w1,decoded)
   row.update(honest_aggregate_residual=honest,honest_rejected=bool(s_bad or p_bad or v_bad),
              raw_proposal_decoded_residual=m.aggregate_relative_error(a0,a1))
   grid=[];v=m.flatten_update(a1);norm=float(np.linalg.norm(v));rng=np.random.default_rng(repeat*100+r);direction=rng.normal(size=len(v));direction/=np.linalg.norm(direction)
   for factor in [.25,.5,.9,1.,1.1,2.,10.]:
    perturbed=v+direction*(factor*1e-6*max(norm,1e-12));candidate={};offset=0
    for key,value in a1.items():
     count=value.numel();candidate[key]=torch.from_numpy(perturbed[offset:offset+count].astype(np.float32)).reshape(value.shape);offset+=count
    error=m.aggregate_relative_error(candidate,a1)
    reject=m.v2_verify_agg(candidate,w1,decoded)
    grid.append(dict(factor=factor,observed_relative_error=error,rejected=bool(reject),within_tolerance=error<=1e-6))
    assert reject==(error>1e-6)
   row['perturbation_grid']=grid
   # A deliberately boundary-adjacent proposal can differ between raw/decoded views.
   if repeat==0 and r==1:
    raw_v=m.flatten_update(a0);delta=v-raw_v;delta_norm=np.linalg.norm(delta)
    if delta_norm>0:
     unit=delta/delta_norm;radius=1e-6*np.linalg.norm(raw_v)
     for amplitude in np.linspace(max(0.,radius-2*delta_norm),radius+2*delta_norm,257):
      x=raw_v+unit*amplitude;candidate={};offset=0
      for key,value in a0.items():
       n=value.numel();candidate[key]=torch.from_numpy(x[offset:offset+n].astype(np.float32)).reshape(value.shape);offset+=n
      e0=m.aggregate_relative_error(candidate,a0);e1=m.aggregate_relative_error(candidate,a1)
      if (e0>1e-6)!=(e1>1e-6):
       boundary.append(dict(repeat=repeat,round=r,raw_residual=e0,decoded_residual=e1,raw_reject=e0>1e-6,decoded_reject=e1>1e-6));break
  all_rows.append(row)
  with (OUT/'rounds.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False,default=lambda v:v.item())+'\n')
  print(json.dumps({k:row[k] for k in ['repeat','round','max_weight_abs_error','exclusion_sets_match','model_relative_error']}),flush=True)
save(OUT/'summary.json',dict(comment=manifest['comment'],repeats=3,rounds_per_repeat=50,round_records=len(all_rows),
    raw_state_matches_training=sum(r['raw_state_matches_training'] for r in all_rows),
    max_decode_abs_error=max(r['max_decode_abs_error'] for r in all_rows),
    max_weight_abs_error=max(r['max_weight_abs_error'] for r in all_rows),
    max_cosine_abs_error=max(r['max_cosine_abs_error'] for r in all_rows),
    max_direction_abs_error=max(r['max_direction_abs_error'] for r in all_rows),
    norm_decision_changes=sum(r['norm_decision_changes'] for r in all_rows),
    cosine_decision_changes=sum(r['cosine_decision_changes'] for r in all_rows),
    direction_decision_changes=sum(r['direction_decision_changes'] for r in all_rows),
    exclusion_matches=sum(r['exclusion_sets_match'] for r in all_rows),
    max_model_relative_error=max(r['model_relative_error'] for r in all_rows),
    max_honest_aggregate_residual=max(r.get('honest_aggregate_residual',0.) for r in all_rows),
    honest_rejections=sum(r.get('honest_rejected',False) for r in all_rows),
    boundary_cases=boundary,scope=manifest['scope']))
print('CKKS sequential replay completed',flush=True)
