"""Server CPU comparison with explicit auxiliary-coverage trade-offs."""
from pathlib import Path
import argparse,copy,ctypes,hashlib,io,json,os,random,secrets,sys,time
import numpy as np
import torch

ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True)
ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path[:0]=[str(a.candidate),str(a.candidate/'core')]
from scripts.exp3 import exp3_governance_tasv2 as m
from sampled_weight_audit import build_admissibility_bounds,verify_sampled_bounds
torch.set_num_threads(2)
CONFIGS={'One complete':['complete'],'Two complete':['complete','complete'],
         'TAS':['Vs','Va_bounds']}
PARAMS=dict(trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,
            refine_anchor=True,ema_alpha=.7,screen_tau=.6,weight_tolerance=1e-6,
            aggregate_tolerance=1e-6,projection_tolerance=.1,sample_ratio=.3)


def native_json(value):
    if isinstance(value,np.generic):return value.item()
    raise TypeError(f'Unsupported observation type: {type(value).__name__}')


def runtime_threads():
    record=dict(torch_intraop=torch.get_num_threads(),torch_interop=torch.get_num_interop_threads(),
       environment={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','PYTHONHASHSEED']},blas_libraries=[])
    for path in (Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.so'):
        try:
            library=ctypes.CDLL(str(path))
            for symbol in ['openblas_get_num_threads64_','openblas_get_num_threads_64_','openblas_get_num_threads']:
                if hasattr(library,symbol):
                    function=getattr(library,symbol);function.restype=ctypes.c_int
                    record['blas_libraries'].append(dict(path=str(path),symbol=symbol,threads=function()));break
        except OSError:continue
    return record


def stamp(call,stages,name):
    start=time.perf_counter_ns();value=call();stages[name]=(time.perf_counter_ns()-start)/1e6;return value


def reference(capture,inputs,context):
    ema,history,anchor,temporal,_,_=copy.deepcopy(capture['prior_trust_state'])
    flat={c:m.flatten_update(u) for c,u in inputs.items()}
    result=m.full_trust(flat,trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,
       refine_anchor=True,ema_weights=ema,ema_alpha=.7,cos_history=history,
       prev_anchor=anchor,temporal_anchor=temporal,return_projection_context=context)
    return result[0],result[-1] if context else None


def binding(capture,inputs,kind):
    hashes={c:m.hash_local_update(u) for c,u in inputs.items()}
    state=m.protocol_state_hash({},*capture['prior_trust_state'])
    record=dict(round=capture['round'],parent=capture['parent_hash'],state=state,
                hashes=sorted(hashes.items()),parameters=PARAMS,kind=kind)
    return hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest(),hashes


def prepare(kind,capture,inputs):
    stages={};start=time.perf_counter_ns()
    if kind=='Va_bounds':
        parent=stamp(lambda:copy.deepcopy(capture['prior_trust_state']),stages,'parent_snapshot')
        bounds=stamp(lambda:build_admissibility_bounds(inputs,parent),stages,'bound_preparation')
        weights=None;context=None
    else:
        weights,context=stamp(lambda:reference(capture,inputs,kind=='Vs'),stages,'full_reference_preparation')
        bounds=None
    key,hashes=stamp(lambda:binding(capture,inputs,kind),stages,'reference_binding')
    stages['total']=(time.perf_counter_ns()-start)/1e6
    return dict(kind=kind,weights=weights,context=context,bounds=bounds,key=key,hashes=hashes),stages


def proposal(capture,inputs,weights,aggregate,round_override=None):
    return dict(weights=weights,aggregate=aggregate,hash=m.proposal_commitment(
        round_number=capture['round'] if round_override is None else round_override,
        parent_hash=capture['parent_hash'],aggregate=aggregate,published_weights=weights,
        update_hashes={c:m.hash_local_update(u) for c,u in inputs.items()}))


def check(local,p,capture,inputs,seed,*,fresh=False):
    stages={};start=time.perf_counter_ns();kind=local['kind']
    if fresh:hashes=local['hashes']
    else:
        key,hashes=stamp(lambda:binding(capture,inputs,kind),stages,'recovery_reference_binding')
        if key!=local['key']:raise ValueError('Invalid local reference binding')
    bound=stamp(lambda:m.proposal_transcript_matches(p['hash'],round_number=capture['round'],
       parent_hash=capture['parent_hash'],aggregate=p['aggregate'],published_weights=p['weights'],
       update_hashes=hashes),stages,'proposal_binding')
    if kind=='Vs':
        badw=stamp(lambda:m.v1_verify_trust(p['weights'],local['weights'],10)[0],stages,'weight_comparison')
        challenge=stamp(lambda:m.make_postcommit_challenge(m.flatten_update(p['aggregate']).size,seed),stages,'challenge_generation')
        bada=stamp(lambda:m.v1_verify_agg(p['aggregate'],p['weights'],inputs,challenge,
                    projection_context=local['context'])[0],stages,'aggregate_crosscheck')
    else:
        if kind=='Va_bounds':
            badw=stamp(lambda:verify_sampled_bounds(p['weights'],local['bounds'],10,
                       42+capture['round']*137)[0],stages,'sampled_weight_comparison')
        elif kind=='Va_exact':
            badw=stamp(lambda:m.v2_verify_trust(p['weights'],local['weights'],10,sample_ratio=.3,
                       seed=42+capture['round']*137)[0],stages,'sampled_weight_comparison')
        else:
            badw=stamp(lambda:any(abs(p['weights'].get(c,0.)-local['weights'].get(c,0.))>1e-6 for c in range(10)),stages,'weight_comparison')
        bada=stamp(lambda:m.v2_verify_agg(p['aggregate'],p['weights'],inputs),stages,'exact_aggregate_check')
    stages['total']=(time.perf_counter_ns()-start)/1e6
    return bound and not badw and not bada,stages


def audit(kinds,p,capture,inputs,seed):
    stages={};locals=[];votes=[];start=time.perf_counter_ns()
    for i,kind in enumerate(kinds):
        local,prep=prepare(kind,capture,inputs);accepted,verify=check(local,p,capture,inputs,seed,fresh=True)
        locals.append(local);votes.append(accepted)
        for k,v in prep.items():stages[f'role_{i+1}_{kind}_prepare_{k}']=v
        for k,v in verify.items():stages[f'role_{i+1}_{kind}_check_{k}']=v
        stages[f'role_{i+1}_{kind}_total']=prep['total']+verify['total']
    stages['total']=(time.perf_counter_ns()-start)/1e6
    return all(votes),locals,stages


def recover(locals,capture,inputs,seed,expected):
    start=time.perf_counter_ns();stages={}
    key,_=stamp(lambda:binding(capture,inputs,locals[0]['kind']),stages,'supplied_weight_binding')
    assert key==locals[0]['key'];weights=dict(locals[0]['weights'])
    aggregate=stamp(lambda:m.weighted_average_updates(inputs,weights),stages,'replacement_aggregation')
    p=stamp(lambda:proposal(capture,inputs,weights,aggregate),stages,'replacement_binding')
    for i,local in enumerate(locals):
        accepted,detail=check(local,p,capture,inputs,seed);assert accepted
        for k,v in detail.items():stages[f'endorser_{i+1}_{local["kind"]}_{k}']=v
    stages['total']=(time.perf_counter_ns()-start)/1e6
    assert m.aggregate_relative_error(aggregate,expected)==0.
    return stages,m.hash_local_update(aggregate)


def tampered_aggregate(aggregate,seed):
    flat=m.flatten_update(aggregate);rng=np.random.default_rng(seed)
    direction=rng.normal(size=flat.size);direction/=np.linalg.norm(direction)
    flat+=direction*(.25*max(np.linalg.norm(flat),1e-12));result={};offset=0
    for k,v in aggregate.items():
        result[k]=torch.from_numpy(flat[offset:offset+v.numel()].astype(np.float32)).reshape(v.shape);offset+=v.numel()
    return result


coverage=[];honest_bounds=[];bounds_times=[];capture_files=[]
for number in range(1,51):
    capture_path=a.base/'results/c2/captured'/f'round_{number:02d}.pt'
    raw=capture_path.read_bytes()
    capture_files.append(dict(round=number,file=capture_path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    capture=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    inputs=capture['local_updates'];locals={kind:prepare(kind,capture,inputs)[0] for kind in ['Vs','Va_exact','Va_bounds','complete']}
    correct=locals['Vs']['weights'];expected=m.weighted_average_updates(inputs,correct)
    bounds_times.append(dict(round=number,stages_ms=locals['Va_bounds']['bounds']['timings_ms']))
    cases={'honest':proposal(capture,inputs,correct,expected)}
    flat={c:m.flatten_update(u) for c,u in inputs.items()}
    # Current attack generator uses these scores only to choose active fallback targets.
    ema,hist,anchor,temporal,_,_=copy.deepcopy(capture['prior_trust_state'])
    _,cosines,_=m.full_trust(flat,trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,
       refine_anchor=True,ema_weights=ema,ema_alpha=.7,cos_history=hist,prev_anchor=anchor,temporal_anchor=temporal)
    for attack in ['A1','A2','A3']:
        published,used,_=m.apply_corrupt_e_attack(correct,cosines,[0,1,2,3],attack,random.Random(42+number),10,
                                                active_client_ids=list(inputs))
        cases[attack]=proposal(capture,inputs,published,m.weighted_average_updates(inputs,used))
        if attack=='A1':
            normalized={c:w/sum(published.values()) for c,w in published.items()}
            cases['A1_normalized']=proposal(capture,inputs,normalized,m.weighted_average_updates(inputs,normalized))
    shifted=dict(correct);source=max(shifted,key=shifted.get)
    target=min((c for c in shifted if c!=source),key=shifted.get);delta=min(.05,shifted[source]/2)
    shifted[source]-=delta;shifted[target]+=delta
    cases['C4_weight_redistribution']=proposal(capture,inputs,shifted,m.weighted_average_updates(inputs,shifted))
    for case,p in cases.items():
        seed=secrets.randbits(128)
        votes={kind:check(local,p,capture,inputs,seed,fresh=True)[0] for kind,local in locals.items()}
        weight_error=max(abs(p['weights'].get(c,0.)-correct.get(c,0.)) for c in range(10))
        residual=m.aggregate_relative_error(p['aggregate'],m.weighted_average_updates(inputs,p['weights']))
        effective=bool(weight_error>1e-6 or residual>1e-6)
        current=votes['Vs'] and votes['Va_exact'];light=votes['Vs'] and votes['Va_bounds']
        assert current==light,(number,case,votes)
        if case=='honest':assert all(votes.values());honest_bounds.append(True)
        if effective:assert not current and not light and not votes['complete']
        coverage.append(dict(round=number,case=case,projection_seed_hex=format(seed,'032x'),sample_seed=42+number*137,weight_error=weight_error,aggregate_residual=residual,
             effective_attack=effective,votes=votes,current_TAS_accepted=current,light_TAS_accepted=light))
    if number%10==0:
        (a.output/'coverage_checkpoint.json').write_text(json.dumps(coverage,indent=2,default=native_json))
        print(json.dumps({'coverage_rounds':number,'proposal_cases':len(coverage)}),flush=True)

observations=[];payloads=[]
for number in [1,3,10,25,50]:
    capture=torch.load(a.base/'results/c2/captured'/f'round_{number:02d}.pt',map_location='cpu',weights_only=False)
    inputs=capture['local_updates'];weights,_=reference(capture,inputs,False)
    expected=m.weighted_average_updates(inputs,weights);good=proposal(capture,inputs,weights,expected)
    bad=proposal(capture,inputs,weights,tampered_aggregate(expected,3100+number))
    payload=io.BytesIO();torch.save(dict(round=number,parent_hash=capture['parent_hash'],inputs=inputs,
         trust_parent=capture['prior_trust_state'],proposal=good),payload)
    payloads.append(dict(round=number,serialized_input_bytes=len(payload.getvalue())))
    for config,kinds in CONFIGS.items():
        assert audit(kinds,good,capture,inputs,secrets.randbits(128))[0]
        accepted,locals,_=audit(kinds,bad,capture,inputs,secrets.randbits(128));assert not accepted
        recover(locals,capture,inputs,secrets.randbits(128),expected)
    jobs=list(CONFIGS)
    for repeat in range(8):
        shift=repeat%len(jobs)
        for config in jobs[shift:]+jobs[:shift]:
            seed=secrets.randbits(128);accepted,_,verify=audit(CONFIGS[config],good,capture,inputs,seed);assert accepted
            accepted,locals,failed=audit(CONFIGS[config],bad,capture,inputs,seed);assert not accepted
            recovery_seed=secrets.randbits(128)
            recovery,state_hash=recover(locals,capture,inputs,recovery_seed,expected)
            observations.append(dict(round=number,configuration=config,repeat=repeat,verification_projection_seed_hex=format(seed,'032x'),
                recovery_projection_seed_hex=format(recovery_seed,'032x'),sample_seed=42+number*137,verification=verify,
                rejected_proposal_audit=failed,recovery=recovery,recovery_aggregate_hash=state_hash,
                closed_loop_ms=failed['total']+recovery['total']))
    (a.output/'timing_checkpoint.json').write_text(json.dumps(observations,indent=2,default=native_json))
    print(json.dumps({'timing_round':number,'timing_observations':len(observations)}),flush=True)

summary={}
for config,kinds in CONFIGS.items():
    rows=[r for r in observations if r['configuration']==config]
    summary[config]={phase:float(np.median([r[phase]['total'] for r in rows])) for phase in ['verification','rejected_proposal_audit','recovery']}
    summary[config]['closed_loop_ms']=float(np.median([r['closed_loop_ms'] for r in rows]))
    summary[config]['logical_received_MiB']=float(np.median([p['serialized_input_bytes']*len(kinds)/2**20 for p in payloads]))
    keys=sorted({k for r in rows for k in r['verification']})
    summary[config]['verification_stages']={k:float(np.median([r['verification'].get(k,0.) for r in rows])) for k in keys}
by_case={}
for case in ['A1','A2','A3','A1_normalized','C4_weight_redistribution']:
    rows=[r for r in coverage if r['case']==case and r['effective_attack']]
    by_case[case]=dict(effective_cases=len(rows),
        rejected={k:sum(not r['votes'][k] for r in rows) for k in ['Vs','Va_exact','Va_bounds','complete']},
        current_TAS_rejected=sum(not r['current_TAS_accepted'] for r in rows),
        light_TAS_rejected=sum(not r['light_TAS_accepted'] for r in rows))
for number in [1,3,10,25,50]:assert len({r['recovery_aggregate_hash'] for r in observations if r['round']==number})==1
result=dict(scope='V_S full trust verification with projection; V_P sampled admissibility with exact aggregation. Single complete and two complete controls use matched reference reuse and fresh recovery endorsements. Frozen-input coverage; serial CPU measurements.',
    host=os.uname().nodename,python=sys.executable,python_version=sys.version,torch=torch.__version__,numpy=np.__version__,parameters=PARAMS,
    runtime_threads=runtime_threads(),captured_inputs=capture_files,
    honest_cases=len(honest_bounds),honest_light_crosscheck_accepted=sum(honest_bounds),
    coverage_summary=by_case,cost_summary=summary,coverage=coverage,observations=observations,
    bounds_stage_profiles=bounds_times,payloads=payloads,
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),a.candidate/'sampled_weight_audit.py',
       a.candidate/'scripts/exp2/exp2_run.py',a.candidate/'scripts/exp3/exp3_governance_tasv2.py']})
(a.output/'lightweight_comparison.json').write_text(json.dumps(result,indent=2,default=native_json))
print(json.dumps(dict(coverage=by_case,costs={k:{n:v for n,v in s.items() if n!='verification_stages'} for k,s in summary.items()}),indent=2),flush=True)
