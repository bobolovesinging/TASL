"""Repair only numeric serialization; reuse completed training and frozen inputs."""
from pathlib import Path
import ast,hashlib,json,sys,numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
OUT=ROOT/'results/comment1'
BASE=learning.compute_tasl_final_weights
def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
    temp.replace(path)
driver.save=save
tree=ast.parse((ROOT/'run_comment1.py').read_text())
node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='replay')
scope=dict(ROOT=ROOT,OUT=OUT,BASE=BASE,driver=driver,np=np,torch=torch)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(ROOT/'run_comment1.py'),'exec'),scope)
scope['replay']()
summaries={name:json.loads((OUT/name/'summary.json').read_text()) for name in ['clean_control','clean_corrected','delayed_control','delayed_corrected']}
def read_rows(name):return json.loads((OUT/name/'round_metrics.json').read_text())
for name,item in summaries.items():
    rows=read_rows(name)
    assert len(rows)==50 and all(not r.get('skipped') for r in rows)
    assert np.mean([r['accuracy'] for r in rows])==item['average_accuracy']
    item['post_onset_average_accuracy']=float(np.mean([r['accuracy'] for r in rows[15:]]))
    item['round16_weights']=json.loads(rows[15]['final_weights'])
    item['round17_weights']=json.loads(rows[16]['final_weights'])
    if name.endswith('corrected'):
        assert item['norm_gate_leak_rounds']==0
clean_a,clean_b=read_rows('clean_control'),read_rows('clean_corrected')
clean_matches=sum(a['state_hash']==b['state_hash'] for a,b in zip(clean_a,clean_b))
assert clean_matches==50
save(OUT/'completed.json',dict(comments=['R4 round2 C1'],summaries=summaries,
    clean_complete_state_matches=clean_matches,
    finalized_by=dict(extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       training_status='Four complete 50-round runs retained; original postprocessing failed to encode numpy int64. This extractor reran only the frozen-input replay.',
       machine_assertions='All completed conditions have zero skipped rounds; corrected conditions have zero norm-rejected weight; clean complete states match 50/50.'),
    interpretation='One seed supports the tested mechanism and records utility; no statistical superiority or arbitrary poisoning guarantee.'))
print(json.dumps(dict(comments=['R4 round2 C1'],clean_complete_state_matches=clean_matches,
    conditions={name:{k:s[k] for k in ['average_accuracy','post_onset_average_accuracy','skipped','norm_gate_leak_rounds','maximum_norm_rejected_weight']} for name,s in summaries.items()})),flush=True)
