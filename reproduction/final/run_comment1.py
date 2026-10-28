"""C1 only: matched clean/amplification runs and captured warm-history replay."""
from pathlib import Path
import copy,json,sys,numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov

BASE=learning.compute_tasl_final_weights
OUT=ROOT/'results/comment1'
OUT.mkdir(parents=True,exist_ok=False)
context=dict(name='',calls=0)
def capture(updates,**kwargs):
    context['calls']+=1
    if context['name']=='delayed_corrected' and context['calls']==16:
        torch.save(dict(updates=updates,kwargs=copy.deepcopy(kwargs),round=16,original_scale=100.),OUT/'warm_history_round16.pt')
    return BASE(updates,**kwargs)
learning.compute_tasl_final_weights=capture

def replay():
    payload=torch.load(OUT/'warm_history_round16.pt',map_location='cpu',weights_only=False)
    kw=payload['kwargs'];kw.pop('gate_current',None);kw.pop('return_history',None)
    original=payload['updates']
    prior=kw['ema_weights'];assert all(prior.get(c,0.)>0 for c in range(4))
    rows=[]
    for construction in ['own_update_amplification','aligned_with_previous_aggregate']:
        for factor in [1.,3.,10.,100.,1000.,10000.]:
            updates={c:v.copy() for c,v in original.items()}
            median=float(np.median([np.linalg.norm(v) for c,v in original.items() if c>=4]))
            ref=kw['temporal_anchor'];unit=ref/np.linalg.norm(ref)
            for c in range(4):
                updates[c]=original[c]/100.*factor if construction=='own_update_amplification' else unit*median*factor
            norms={c:float(np.linalg.norm(v)) for c,v in updates.items()}
            limit=2.5*float(np.median(list(norms.values())))
            for corrected in [False,True]:
                w,_,_=BASE(updates,gate_current=corrected,**kw)
                aggregate=sum(w[c]*v for c,v in updates.items())
                bad=[c for c,n in norms.items() if n>limit]
                row=dict(construction=construction,factor=factor,corrected=corrected,
                         median_norm=limit/2.5,aggregate_norm=float(np.linalg.norm(aggregate)),
                         aggregate_bound=limit,malicious_weight=sum(w[c] for c in range(4)),
                         norm_rejected_weight=sum(w[c] for c in bad),norm_rejected_clients=bad,
                         normalized_weight_sum=sum(w.values()),positive_historical_clients=sum(prior.get(c,0.)>0 for c in range(4)))
                if corrected:
                    assert row['norm_rejected_weight']==0.
                    assert row['aggregate_norm']<=limit+1e-10
                rows.append(row)
    driver.save(OUT/'warm_history_replay.json',dict(source_round=16,
        captured_inputs_sha256=driver.sha(OUT/'warm_history_round16.pt'),rows=rows,
        purpose='R4 second-round Comment 1: amplification bound with strictly positive accumulated trust'))

torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
# Tiny checks include the shared corrected weight rule and actual learning entry point.
context.update(name='smoke',calls=0)
ref,clients,test=driver.run_one(OUT,'smoke_reference',True,'noexecutor','label_flip',3,smoke=32)
driver.run_one(OUT,'smoke_audited',True,'trust_v1v2','label_flip',3,smoke=32)
driver.compare(OUT,'smoke_reference','smoke_audited')
driver.learning_smoke(OUT,clients,test,ref)

for name,corrected,attack in [
    ('clean_control',False,'none'),('clean_corrected',True,'none'),
    ('delayed_control',False,'scaling'),('delayed_corrected',True,'scaling')]:
    context.update(name=name,calls=0)
    driver.run_one(OUT,name,corrected,'noexecutor',attack,50)
    assert context['calls']==50
replay()
summaries={name:json.loads((OUT/name/'summary.json').read_text()) for name in ['clean_control','clean_corrected','delayed_control','delayed_corrected']}
driver.save(OUT/'completed.json',dict(comments=['R4 round2 C1'],summaries=summaries,
    source_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in sorted(ROOT.rglob('*.py'))},
    interpretation='Read with norm-gate leakage, skip counts and utility changes; one seed is a targeted mechanism check.'))
print(json.dumps(dict(state='completed',comments=['R4 round2 C1'],summaries=summaries)),flush=True)
