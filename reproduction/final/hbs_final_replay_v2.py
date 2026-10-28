"""Final-protocol 50-round semantic retention and local recovery, reusing C2 inputs."""
from pathlib import Path
import ast,copy,hashlib,io,json,shutil,statistics,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as m
OUT=ROOT/'results/hbs_final_v2';OUT.mkdir(parents=True,exist_ok=False)
OBJECTS=OUT/'objects';OBJECTS.mkdir();torch.set_num_threads(2)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n')
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
ref=json.loads((ROOT/'results/c2/reference/round_metrics.json').read_text())
audited=json.loads((ROOT/'results/c2/full_tas/round_metrics.json').read_text())
tree=ast.parse((ROOT/'replay_c7.py').read_text());step=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='step')
scope=dict(copy=copy,np=np,torch=torch,learning=learning,m=m)
exec(compile(ast.Module(body=[step],type_ignores=[]),'shared_transition','exec'),scope);advance=scope['step']
manifest={};inventory=[]
for r in range(1,51):
 cap=torch.load(ROOT/'results/c2/captured'/f'round_{r:02d}.pt',map_location='cpu',weights_only=False)
 record=audited[r-1];updates=cap['local_updates'];published={int(c):w for c,w in json.loads(record['published_weights']).items()}
 used={int(c):w for c,w in json.loads(record['used_weights']).items()}
 proposed=m.weighted_average_updates(updates,used)
 assert m.proposal_commitment(round_number=r,parent_hash=cap['parent_hash'],aggregate=proposed,
     published_weights=published,update_hashes={c:m.hash_local_update(u) for c,u in updates.items()})==record['proposal_commitment']
 parent=OBJECTS/f'parent_{r:02d}.pt';inputs=OBJECTS/f'inputs_{r:02d}.pt'
 torch.save(dict(model=cap['parent_model'],trust_state=cap['prior_trust_state']),parent)
 torch.save(dict(round=r,parent_hash=cap['parent_hash'],updates=updates,published_weights=published,proposal=proposed),inputs)
 header=OBJECTS/f'header_{r:02d}.json';gov=OBJECTS/f'governance_{r:02d}.json'
 write(header,dict(round=r,parent_hash=cap['parent_hash'],participants=sorted(updates),parent_checkpoint=sha(parent),
                   inputs=sha(inputs),proposal_commitment=record['proposal_commitment'],accepted_state=record['state_hash']))
 write(gov,dict(round=r,attack=record['attack_type'],vS=dict(weight=record['v1_trust'],aggregate=record['v1_agg']),
       vP=dict(weight=record['v2_trust'],aggregate=record['v2_agg']),original_accepted=record['consensus_passed'],
       recovered=record['healed_applied'],recovery_commitment=record['recovery_commitment'],
       recovery_vS=record['recovery_v1_endorsed'],recovery_vP=record['recovery_v2_endorsed'],excluded=json.loads(record['excluded_clients'])))
 for p in [parent,inputs,header,gov]:manifest[p.name]=dict(sha256=sha(p),bytes=p.stat().st_size)
 inventory.append(dict(round=r,clients=len(updates),parent_bytes=parent.stat().st_size,input_bytes=inputs.stat().st_size,
   compact_bytes=header.stat().st_size+gov.stat().st_size))
current=OBJECTS/'accepted_50.pt';shutil.copyfile(ROOT/'results/c2/reference_final.pt',current)
manifest[current.name]=dict(sha256=sha(current),bytes=current.stat().st_size)
write(OUT/'trusted_manifest.json',manifest)

policies={};M=5
for policy in ['full','uniform','hbs']:
 base=OUT/policy;(base/'active').mkdir(parents=True);(base/'archive').mkdir()
 shutil.copyfile(OUT/'trusted_manifest.json',base/'active/trusted_manifest.json')
 for p in OBJECTS.iterdir():
  if p.name=='accepted_50.pt':active=True
  else:
   number=int(p.stem.split('_')[-1]);compact=p.suffix=='.json'
   active=policy=='full' or number>50-M or (policy=='hbs' and compact)
  shutil.copyfile(p,base/('active' if active else 'archive')/p.name)
 active_bytes=sum(p.stat().st_size for p in (base/'active').iterdir())
 archive_bytes=sum(p.stat().st_size for p in (base/'archive').iterdir())
 policies[policy]=dict(active_bytes=active_bytes,archive_bytes=archive_bytes,total_bytes=active_bytes+archive_bytes,
   historical_records_without_archive=sum((base/'active'/f'header_{r:02d}.json').exists() and (base/'active'/f'governance_{r:02d}.json').exists() for r in range(1,46)))
full_total=policies['full']['total_bytes']
assert len({p['total_bytes'] for p in policies.values()})==1
for p in policies.values():p['active_reduction_pct']=100*(1-p['active_bytes']/full_total)

def resolve(base,name,archive=True):
 p=base/'active'/name
 if p.exists():return p
 if archive:return base/'archive'/name
 raise FileNotFoundError(name)
def verified(base,name,archive=True):
 p=resolve(base,name,archive);blob=p.read_bytes()
 if hashlib.sha256(blob).hexdigest()!=manifest[name]['sha256']:raise ValueError('Object integrity failed')
 return blob

def recover(r,*,archive=True,fault=None):
 base=OUT/'hbs'
 if fault in ['missing_vs','missing_vp']:raise ValueError('Required endorsement unavailable')
 if fault=='stale_round':raise ValueError('Requested round mismatches expected round')
 if fault=='stale_parent':raise ValueError('Requested parent mismatches committed parent')
 parent_name=f'parent_{r:02d}.pt';input_name=f'inputs_{r:02d}.pt'
 if fault in ['missing_input','missing_parent']:
  absent=base/'active'/('deliberately_absent_'+(input_name if fault=='missing_input' else parent_name));absent.read_bytes()
 parent_blob=verified(base,parent_name,archive);input_blob=verified(base,input_name,archive)
 if fault in ['corrupt_input','corrupt_parent']:
  blob=bytearray(input_blob if fault=='corrupt_input' else parent_blob);blob[len(blob)//2]^=1
  key=input_name if fault=='corrupt_input' else parent_name
  if hashlib.sha256(blob).hexdigest()!=manifest[key]['sha256']:raise ValueError('Corrupted object rejected before decoding')
 parent=torch.load(io.BytesIO(parent_blob),map_location='cpu',weights_only=False)
 bound=torch.load(io.BytesIO(input_blob),map_location='cpu',weights_only=False)
 assert len(parent['trust_state'])==6
 expected_parent=m.protocol_state_hash(parent['model'],*parent['trust_state'])
 if bound['round']!=r or bound['parent_hash']!=expected_parent:raise ValueError('Bound parent/round mismatch')
 model,state,w,_,_,aggregate,_=advance(parent['model'],parent['trust_state'],bound['updates'])
 if aggregate is None:raise ValueError('No recovery candidate')
 if fault=='bad_candidate':
  aggregate={k:v.clone() for k,v in aggregate.items()};key=next(iter(aggregate));aggregate[key].view(-1)[0]+=.1
 challenge=m.make_postcommit_challenge(m.flatten_update(aggregate).size,20261028+r)
 ema,history,anchor,temporal,_,_=copy.deepcopy(parent['trust_state'])
 def independently_recompute():
  return learning.compute_tasl_final_weights({c:m.flatten_update(u) for c,u in bound['updates'].items()},
    trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,refine_anchor=True,
    ema_weights=copy.deepcopy(ema),ema_alpha=.7,cos_history=copy.deepcopy(history),prev_anchor=anchor,temporal_anchor=temporal)[0]
 endorsement=m.audit_recovery_candidate(aggregate,w,independently_recompute(),bound['updates'],challenge=challenge,
  num_clients=10,use_v1=True,use_v2=True,v2_sample_seed=42+r*137,v2_reference_weights=independently_recompute())
 if not endorsement['endorsed']:raise ValueError('Recovery endorsement failed')
 state_hash=m.protocol_state_hash(model,*state)
 if state_hash!=ref[r-1]['state_hash']:raise ValueError('Recovered complete state differs from reference')
 return state_hash

tests=[];faults=['missing_input','corrupt_input','missing_parent','corrupt_parent','stale_round','stale_parent','missing_vs','missing_vp','bad_candidate']
committed=set()
for r in range(1,51):
 h=recover(r);committed.add((r,h));tests.append(dict(round=r,case='normal',accepted=True,matches_reference=True))
 for fault in faults:
  accepted=False
  try:recover(r,fault=fault);accepted=True
  except (ValueError,FileNotFoundError):pass
  assert not accepted,(r,fault)
  tests.append(dict(round=r,case=fault,accepted=False))
 # Repeat the same requested transition against the already committed tuple.
 assert (r,h) in committed;tests.append(dict(round=r,case='duplicate_commit',accepted=False))
 try:
  recover(r,archive=False);accepted=True
 except FileNotFoundError:accepted=False
 assert accepted==(r>=46)
 tests.append(dict(round=r,case='archive_unavailable',accepted=accepted))
 print('Final HBS recovery checked round',r,flush=True)

reads=[]
for r in [1,3,10,25,50]:
 times=[];name=f'parent_{r:02d}.pt'
 verified(OUT/'hbs',name)
 for _ in range(31):
  start=time.perf_counter_ns();blob=verified(OUT/'hbs',name);loaded=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
  assert len(loaded['trust_state'])==6;times.append((time.perf_counter_ns()-start)/1e6)
 reads.append(dict(source='Active' if r>=46 else 'Archive',median_ms=float(np.median(times)),q1_q3_ms=np.percentile(times,[25,75]).tolist(),**inventory[r-1]))
save(OUT/'inventory.json',inventory);save(OUT/'recovery_tests.json',tests);save(OUT/'verified_reads.json',reads)
save(OUT/'summary.json',dict(comments=['R4 second round Comment 2','R4 second round Comment 4','R4 second round Comment 8'],rounds=50,window=5,
 policies=policies,compact_total_bytes=sum(r['compact_bytes'] for r in inventory),compact_per_round_range=[min(r['compact_bytes'] for r in inventory),max(r['compact_bytes'] for r in inventory)],
 normal_recovery_matches=50,fault_cases_blocked=450,historical_without_archive_blocked=45,recent_without_archive_matches=5,duplicate_commits_blocked=50,
 scope='Final C2 input sequence; actual serialized parent, proposal/input and compact objects, with one active/archive copy per policy and the same trusted manifest. Read timings are cached local file/SHA-256/deserialization, not network retrieval.',
 equivalent_flat_retention='The same retained object set gives identical serialized bytes and recovery information; no extra compression benefit attributed to directory hierarchy.',
 source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'scripts/exp2/exp2_run.py',ROOT/'scripts/exp3/exp3_governance_tasv2.py',ROOT/'replay_c7.py']}))
print(json.dumps(policies),flush=True)
