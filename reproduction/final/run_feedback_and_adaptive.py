"""Final-policy feedback training: three CKKS executions and one white-box stress test."""
from pathlib import Path
import ast,copy,hashlib,json,sys,time
import numpy as np,torch,tenseal as ts
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as m
import run_experiments as driver
from final_input_adapter import build
run,generated=build();OUT=ROOT/'results/feedback';OUT.mkdir(parents=True,exist_ok=False)
(OUT/'instrumented_runner.py').write_text(generated)
torch.set_num_threads(3);torch.use_deterministic_algorithms(True)
def save(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False,default=lambda x:x.item())+'\n')
scope=dict(np=np,torch=torch,ts=ts,m=m,time=time)
tree=ast.parse((ROOT/'replay_c7.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='roundtrip')
exec(compile(ast.Module(body=[node],type_ignores=[]),'shared_roundtrip','exec'),scope)
roundtrip=scope['roundtrip']
def execute(name,transform,attack='label_flip',rounds=50,smoke=0):
    clients,test,_=driver.loaders(smoke)
    m.full_trust=learning.compute_tasl_final_weights
    result=run('trust_v1v2',clients,test,rounds,driver.schedule(rounds),torch.device('cuda'),42,
        local_epochs=1 if smoke else 2,lr=.02,momentum=0.,weight_decay=0.,byzantine_clients=list(range(4)),
        ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,enable_exclusion=True,exclusion_budget=4,
        client_attack=attack,input_transform=transform,round_record_dir=str(OUT/name))
    rows=result.round_logs;save(OUT/(name+'.json'),rows);return rows
# Verify a transparent external-input hook leaves the entire state transition unchanged.
clients,test,_=driver.loaders(32)
original=m.run_experiment_group('trust_v1v2',clients,test,3,driver.schedule(3),torch.device('cuda'),42,
    local_epochs=1,lr=.02,momentum=0.,weight_decay=0.,byzantine_clients=list(range(4)),ema_alpha=.7,
    trust_power=5.,max_weight_ratio=1.5,enable_exclusion=True,exclusion_budget=4,client_attack='label_flip')
transparent=execute('transparent_hook',lambda r,p,s,u:u,rounds=3,smoke=32)
assert all(a['state_hash']==b['state_hash'] for a,b in zip(original.round_logs,transparent))
reference=json.loads((ROOT/'results/c2/full_tas/round_metrics.json').read_text())
all_runs=[]
for repeat in range(3):
    private=ts.context(ts.SCHEME_TYPE.CKKS,8192,coeff_mod_bit_sizes=[60,40,40,60],n_threads=2)
    private.global_scale=2**40
    # Updates are encrypted with a context that cannot decrypt.
    public=ts.context_from(private.serialize(save_secret_key=False),n_threads=2)
    errors=[]
    def transform(r,parent,state,updates):
        decoded={};maximum=0.;size=0;started=time.monotonic()
        for c,update in updates.items():
            raw=m.flatten_update(update);parts=[]
            for start in range(0,len(raw),4096):
                segment=raw[start:start+4096];pad=np.zeros(4096);pad[:len(segment)]=segment
                blob=ts.ckks_vector(public,pad.tolist()).serialize();size+=len(blob)
                values=np.asarray(ts.ckks_vector_from(private,blob).decrypt())[:len(segment)]
                maximum=max(maximum,float(np.max(np.abs(values-segment))));parts.append(values.astype(np.float32))
            vector=np.concatenate(parts);offset=0;decoded[c]={}
            for key,tensor in update.items():
                count=tensor.numel();decoded[c][key]=torch.from_numpy(vector[offset:offset+count].copy()).reshape(tensor.shape);offset+=count
        errors.append(dict(round=r,max_decode_abs=maximum,ciphertext_bytes=size,seconds=time.monotonic()-started))
        return decoded
    rows=execute(f'ckks_{repeat}',transform)
    attack=[x for x in rows if x['attack_active']];honest=[x for x in rows if not x['attack_active']]
    rejection=lambda x:any(x[k] for k in ['v1_trust','v1_agg','v2_trust','v2_agg'])
    assert all(x['recovery_state_hash']==x['state_hash'] for x in rows if x.get('healed_applied') and 'recovery_state_hash' in x)
    summary=dict(repeat=repeat,avg=float(np.mean([x['accuracy'] for x in rows])),final=rows[-1]['accuracy'],
        honest_acceptance=sum(not rejection(x) for x in honest),honest_count=len(honest),
        attacked_rejected=sum(rejection(x) for x in attack),attacked_count=len(attack),
        max_decode_abs=max(x['max_decode_abs'] for x in errors),
        exclusion_sets_equal_plain=sum(x['excluded_clients']==y['excluded_clients'] for x,y in zip(rows,reference)),
        skipped=sum(x.get('skipped',False) for x in rows))
    save(OUT/f'crypto_errors_{repeat}.json',errors);save(OUT/f'ckks_summary_{repeat}.json',summary)
    all_runs.append(summary);print(json.dumps(summary),flush=True)
save(OUT/'ckks_summary.json',dict(repeats=3,training_feedback=True,runs=all_runs,
    scope='Single-key CKKS encode/decode; decoded global model drives subsequent local training. Independent keys, same data seed42. Not distributed threshold decryption.'))
# Defence-aware coordinated poisoning: optimize deviation from the honest aggregate,
# using the prescribed defence as an oracle, without selecting against test accuracy.
probes=[]
def stress(r,parent,state,updates):
    if r<16:return updates
    ids=[c for c in updates if c<4]
    if not ids:return updates
    honest={c:u for c,u in updates.items() if c>=4}
    center=m.weighted_average_updates(honest,{c:1/len(honest) for c in honest})
    h=m.flatten_update(center);h=h/max(np.linalg.norm(h),1e-12)
    anchor=state[3] if state[3] is not None else h
    anchor=anchor/max(np.linalg.norm(anchor),1e-12)
    median=float(np.median([np.linalg.norm(m.flatten_update(u)) for u in honest.values()]))
    best=None
    for mix in [0.,.25,.5,.75,1.]:
        direction=(1-mix)*anchor-mix*h;direction/=max(np.linalg.norm(direction),1e-12)
        for scale in [.5,1.,1.5]:
            v=direction*(scale*median);candidate={};offset=0
            for key,tensor in updates[ids[0]].items():
                n=tensor.numel();candidate[key]=torch.from_numpy(v[offset:offset+n].astype(np.float32).copy()).reshape(tensor.shape);offset+=n
            trial={c:(copy.deepcopy(candidate) if c in ids else u) for c,u in updates.items()}
            flat={c:m.flatten_update(u) for c,u in trial.items()}
            weights=learning.compute_tasl_final_weights(flat,trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,
                refine_anchor=True,ema_weights=state[0],ema_alpha=.7,cos_history=state[1],prev_anchor=state[2],temporal_anchor=state[3])[0]
            aggregate=m.weighted_average_updates(trial,weights)
            objective=float(np.linalg.norm(m.flatten_update(aggregate)-m.flatten_update(center)))
            if best is None or objective>best[0]:best=(objective,trial,mix,scale,sum(weights.get(c,0.) for c in ids))
    probes.append(dict(round=r,candidates=15,mix=best[2],norm_multiple=best[3],deviation=best[0],malicious_weight=best[4]))
    return best[1]
rows=execute('defence_aware',stress,attack='none')
save(OUT/'defence_aware_probes.json',probes)
save(OUT/'defence_aware_summary.json',dict(avg=float(np.mean([x['accuracy'] for x in rows])),final=rows[-1]['accuracy'],
    attack_onset=16,attacked_rounds=len(probes),max_malicious_weight=max(x['malicious_weight'] for x in probes),
    scope='White-box coordinated update selection with honest-update and defence-state access; 15 candidates per attacked round maximize aggregate deviation, not test loss. One data seed, no baseline ranking.'))
save(OUT/'completed.json',dict(source_sha256={p.name:driver.sha(p) for p in [Path(__file__),ROOT/'final_input_adapter.py',ROOT/'scripts/exp2/exp2_run.py',ROOT/'scripts/exp3/exp3_governance_tasv2.py']},transparent_hook_complete_matches=3))
