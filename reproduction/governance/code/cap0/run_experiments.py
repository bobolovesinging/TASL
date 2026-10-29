"""Round-2 gate and complete-state consistency probes on existing Fashion-MNIST."""
import argparse, functools, hashlib, json, os, platform, sys, time
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov

def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

CONFIG=dict(seed=42,clients=10,byzantine_ratio=.4,alpha=.3,rounds=50,
    local_epochs=2,batch_size=128,lr=.02,momentum=0.,weight_decay=0.,
    ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,screen_tau=.6,
    norm_gate_multiple=2.5,min_cos_threshold=.2,enable_exclusion=True,
    delayed_start_round=16,delayed_scaling_factor=100.,
    v1_weight_tolerance=gov.TOLERANCE,v1_projection_tolerance=gov.DEFAULT_PROJECTION_TOLERANCE,
    v2_aggregate_tolerance=gov.AGGREGATE_TOLERANCE,v2_sample_client_ratio=.3)

def loaders(smoke=0):
    gov.set_seed(CONFIG['seed'])
    clients,test=gov.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=42,dataset='fashionmnist')
    inventory=[]
    for c,l in enumerate(clients):
        labels=l.dataset.labels.cpu().numpy()
        inventory.append(dict(client=c,samples=len(labels),labels_sha256=hashlib.sha256(labels.tobytes()).hexdigest(),
                              data_sha256=hashlib.sha256(l.dataset.data.cpu().numpy().tobytes()).hexdigest()))
    if smoke:
        clients=[DataLoader(Subset(l.dataset,range(min(smoke,len(l.dataset)))),batch_size=16,shuffle=True) for l in clients]
        test=DataLoader(Subset(test.dataset,range(64)),batch_size=32,shuffle=False)
    return clients,test,inventory

def schedule(rounds):
    sched=gov.make_chaos_schedule(rounds,seed=2026,executor_attack='mixed12',attacks_per_block=6)
    # Include all three Executor tampering types, cycling independently of data.
    for index,r in enumerate(sorted(r for r,a in sched.items() if a.active)):
        sched[r].attack_type=['A1','A2','A3'][index%3]
    return sched

def run_one(out,name,gate,group,attack,rounds,smoke=0,extra=None):
    dest=out/name
    if dest.exists():raise FileExistsError(dest)
    dest.mkdir(parents=True)
    clients,test,inventory=loaders(smoke)
    sched=schedule(rounds)
    gov.full_trust=functools.partial(learning.compute_tasl_final_weights,gate_current=gate)
    manifest=dict(name=name,comments=['R4 round2 C2'],gate_current=gate,
        group=group,client_attack=attack,configuration=CONFIG,rounds=rounds,smoke_samples_per_client=smoke,extra=extra or {},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(ROOT.rglob('*.py'))},
        partition=inventory,attack_schedule={r:asdict(a) for r,a in sched.items()},
        environment=dict(torch=torch.__version__,numpy=np.__version__,python=sys.version,host=platform.node(),
                         gpu=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES')),
        scope='Single-process training/audit/recovery; not distributed consensus or threshold decryption')
    save(dest/'manifest.json',manifest)
    started=time.monotonic()
    result=gov.run_experiment_group(group,clients,test,rounds,sched,torch.device('cuda'),42,
        local_epochs=1 if smoke else 2,lr=.02,momentum=0.,weight_decay=0.,
        byzantine_clients=list(range(4)),ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,
        enable_exclusion=True,client_attack=attack,
        client_attack_start=16 if attack=='scaling' else 1,client_scaling_factor=100.,
        round_record_dir=str(dest/'rounds'),**(extra or {}))
    rows=result.round_logs
    for row in rows:
        if row.get('skipped'):continue
        weights={int(k):v for k,v in json.loads(row['final_weights']).items()}
        norms={int(k):v for k,v in json.loads(row['update_norms']).items()}
        median=float(np.median(list(norms.values())))
        gated=[c for c,n in norms.items() if n>2.5*median]
        row['norm_rejected_clients']=gated
        row['norm_rejected_weight']=sum(weights.get(c,0.) for c in gated)
        row['malicious_client_weight']=sum(weights.get(c,0.) for c in range(4)) if attack!='none' else 0.
        if gate:
            assert row['norm_rejected_weight']==0., 'Current norm gate violated'
            assert abs(sum(weights.values())-1.)<1e-12
    save(dest/'round_metrics.json',rows)
    honest=[r for r in rows if not r.get('skipped') and not r['attack_active']]
    attacked=[r for r in rows if not r.get('skipped') and r['attack_active']]
    rejected=lambda r:bool(r['v1_trust'] or r['v1_agg'] or r['v2_trust'] or r['v2_agg'])
    tp=sum(rejected(r) for r in attacked);fp=sum(rejected(r) for r in honest)
    summary=dict(name=name,comments=manifest['comments'],rounds_completed=len(rows),
        average_accuracy=float(np.mean(result.accuracy_history)),best_accuracy=float(max(result.accuracy_history)),
        seconds=time.monotonic()-started,skipped=sum(r.get('skipped',0) for r in rows),
        norm_gate_leak_rounds=sum(r.get('norm_rejected_weight',0.)>0 for r in rows),
        maximum_norm_rejected_weight=max(r.get('norm_rejected_weight',0.) for r in rows),
        honest_proposals=len(honest),attacked_proposals=len(attacked),true_rejections=tp,false_rejections=fp,
        honest_acceptance=(len(honest)-fp)/len(honest) if honest else None,
        precision=tp/(tp+fp) if tp+fp else None,recall=tp/len(attacked) if attacked else None,
        interpretation='Run finished; scope and pairing checks must still be assessed before manuscript use.')
    save(dest/'summary.json',summary)
    print(json.dumps(summary),flush=True)
    return rows,clients,test

def compare(out,a,b):
    left=json.loads((out/a/'round_metrics.json').read_text())
    right=json.loads((out/b/'round_metrics.json').read_text())
    matches=[x['state_hash']==y['state_hash'] for x,y in zip(left,right)]
    result=dict(reference=a,audited=b,rounds=len(matches),matching_complete_states=sum(matches),
                mismatched_rounds=[i+1 for i,v in enumerate(matches) if not v],
                identical_accuracies=all(x['accuracy']==y['accuracy'] for x,y in zip(left,right)),
                fully_paired=len(left)==len(right) and all(matches))
    save(out/'paired_consistency.json',result)
    print(json.dumps(result),flush=True)
    assert result['fully_paired'], 'Paired complete-state consistency failed; do not launch further experiments'

def learning_smoke(out,clients,test,reference):
    # Exercise the actual learning entry point, not just an alias comparison.
    config=dict(experiment=dict(seed=42),training=dict(local_epochs=1,lr=.02,momentum=0.,weight_decay=0.),
        model=dict(num_classes=10),algorithm_params=dict(tasl=dict(trust_power=5.,weight_cap_ratio=1.5,
            ema_alpha=.7,norm_penalty=.8,fusion='multiplicative',exclusion=True,screen_tau=.6)))
    result=learning.run_single('tasl_full','label_flip',clients,test,config,gov.FedAvgCNN,10,4,
                               len(reference),torch.device('cuda'))
    rows=result['round_metrics']
    save(out/'learning_smoke.json',rows)
    matching=[x['state_hash']==y['state_hash'] for x,y in zip(reference,rows)]
    save(out/'learning_governance_consistency.json',dict(matching=matching,rounds=len(rows)))
    assert len(rows)==len(reference) and all(matching), 'Learning and governance states diverged'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['smoke','full'],required=True)
    args=ap.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    out=ROOT/'results'/args.phase
    out.mkdir(parents=True,exist_ok=True)
    if args.phase=='smoke':
        ref,clients,test=run_one(out,'lf_reference',True,'noexecutor','label_flip',3,smoke=32)
        run_one(out,'lf_full_tas',True,'trust_v1v2','label_flip',3,smoke=32)
        compare(out,'lf_reference','lf_full_tas')
        learning_smoke(out,clients,test,ref)
    else:
        checks=json.loads((ROOT/'results/smoke/paired_consistency.json').read_text())
        learning_checks=json.loads((ROOT/'results/smoke/learning_governance_consistency.json').read_text())
        assert checks['fully_paired'] and all(learning_checks['matching'])
        # Priority order: C1 failure and fix, then C2 recovery pairing, then utility.
        for name,gate,group,attack in [
            ('delayed_amplification_legacy',False,'noexecutor','scaling'),
            ('delayed_amplification_corrected',True,'noexecutor','scaling'),
            ('lf_reference',True,'noexecutor','label_flip'),
            ('lf_full_tas',True,'trust_v1v2','label_flip'),
            ('clean_legacy',False,'noexecutor','none'),
            ('clean_corrected',True,'noexecutor','none')]:
            run_one(out,name,gate,group,attack,50)
            if name=='lf_full_tas':compare(out,'lf_reference','lf_full_tas')
    save(out/'completed.json',dict(phase=args.phase,finished_utc=time.time()))

if __name__=='__main__':main()
