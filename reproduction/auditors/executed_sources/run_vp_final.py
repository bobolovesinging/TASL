from pathlib import Path
import argparse,collections,functools,hashlib,json,os,platform,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov
ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);a=ap.parse_args()
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
gov.full_trust=functools.partial(learning.compute_tasl_final_weights,gate_current=True)
SOURCES={str(p.relative_to(ROOT)):driver.sha(p) for p in ROOT.rglob('*.py')}
OUT=ROOT/'results'/a.dataset;OUT.mkdir(parents=True,exist_ok=False)
def save(p,x):driver.save(p,x)
def run(attack,cap,group):
 label='full_tas' if group=='trust_v1v2' else 'vp_only';dest=OUT/f'cap{cap}'/attack/label;dest.mkdir(parents=True,exist_ok=False)
 gov.set_seed(42);clients,test=gov.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=42,dataset=a.dataset)
 inv=[dict(client=i,samples=len(l.dataset),labels_sha256=hashlib.sha256(l.dataset.labels.cpu().numpy().tobytes()).hexdigest(),data_sha256=hashlib.sha256(l.dataset.data.cpu().numpy().tobytes()).hexdigest()) for i,l in enumerate(clients)]
 old=Path('/home/njit/round2_C2_fixed_v2_20261029/results')/a.dataset/'matrix'/a.dataset/attack if cap==0 else Path('/home/njit/round2_C2_matrix_20261029/results/matrix')/a.dataset/attack
 reference=json.loads((old/'reference/round_metrics.json').read_text())
 original_manifest=json.loads((old/'vp_only/manifest.json').read_text())
 assert inv==original_manifest['partition'],'Partition differs from original experiment'
 config=dict(dataset=a.dataset,attack=attack,cap=cap,group=label,seed=42,rounds=50,alpha=.3,client_num=10,byzantine_clients=[0,1,2,3],local_epochs=2,batch_size=128,lr=.02,momentum=0.,weight_decay=0.,ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,screen_tau=.6,client_scaling_factor=5.,client_attack_start=1,vp_rule='sampled_admissibility_bounds',sample_ratio=.3)
 save(dest/'manifest.json',dict(configuration=config,partition=inv,source_sha256=SOURCES,original_manifest_sha256=driver.sha(old/'vp_only/manifest.json'),reference_sha256=driver.sha(old/'reference/round_metrics.json'),executor_schedule={r:vars(x) for r,x in driver.schedule(50).items()},environment=dict(host=platform.node(),python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),gpu_index=os.getenv('CUDA_VISIBLE_DEVICES'),threads=torch.get_num_threads())))
 start=time.monotonic()
 result=gov.run_experiment_group(group,clients,test,50,driver.schedule(50),torch.device('cuda'),42,local_epochs=2,lr=.02,momentum=0.,weight_decay=0.,byzantine_clients=[0,1,2,3],ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,enable_exclusion=cap>0,exclusion_budget=cap,client_attack=attack,client_attack_start=1,client_scaling_factor=5.,round_record_dir=str(dest/'rounds'),final_state_path=str(dest/'final_state.pt'))
 rows=result.round_logs;save(dest/'round_metrics.json',rows)
 rejected=lambda r:any(r.get(k,False) for k in ['v1_trust','v1_agg','v2_trust','v2_agg'])
 att=[r for r in rows if r['attack_active']];hon=[r for r in rows if not r['attack_active']]
 tp=sum(rejected(r) for r in att);fp=sum(rejected(r) for r in hon)
 effective=sum(max([abs(json.loads(r['published_weights']).get(c,0)-v) for c,v in json.loads(r['final_weights']).items()]+[0])>1e-6 or r['proposal_relative_error']>1e-6 for r in att)
 assert len(rows)==50 and len(att)==30 and len(hon)==20 and effective==30
 assert not any(r.get('skipped') for r in rows)
 assert all(r['recovery_endorsed'] and r['healed_applied'] and r['recovery_consistent'] for r in rows if rejected(r))
 if cap==0:assert not any(json.loads(r['excluded_clients']) for r in rows)
 state_matches=sum(x['state_hash']==y['state_hash'] for x,y in zip(rows,reference))
 if group=='trust_v1v2':assert tp==30 and fp==0 and state_matches==50,'Full TAS reference-state regression'
 summary=dict(**config,seconds=time.monotonic()-start,avg_accuracy=float(np.mean([r['accuracy'] for r in rows])),best_accuracy=max(r['accuracy'] for r in rows),attacks=30,honest=20,effective_attacks=effective,true_rejections=tp,false_rejections=fp,honest_accepted=20-fp,precision=tp/(tp+fp) if tp+fp else None,recall=tp/30,state_matches=state_matches,accuracy_matches=sum(x['accuracy']==y['accuracy'] for x,y in zip(rows,reference)),skipped=0,recovery_validations=tp+fp,attack_types=dict(collections.Counter(r['attack_type'] for r in att)))
 save(dest/'summary.json',summary);print(json.dumps(summary),flush=True)
for cap in [0,4]:
 for attack in ['label_flip','scaling']:
  run(attack,cap,'trust_v2')
  if cap==0:run(attack,cap,'trust_v1v2')
assert SOURCES=={str(p.relative_to(ROOT)):driver.sha(p) for p in ROOT.rglob('*.py')}
save(OUT/'completed.json',dict(completed=True,source_unchanged=True,conditions=6))
