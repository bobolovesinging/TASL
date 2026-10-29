"""Final-kernel MNIST/Fashion-MNIST matrix, 40% clients, fixed participation; permanent exclusion disabled in every group."""
from pathlib import Path
import argparse,collections,datetime,functools,hashlib,json,os,shutil,subprocess,sys,time
import numpy as np,torch
from torch.utils.data import DataLoader,Subset
from torchvision import datasets
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov
CONFIG=json.loads((ROOT/'matrix_config.json').read_text())
ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True,choices=['mnist','fashionmnist']);ARGS=ap.parse_args()
OUT=ROOT/'results'/ARGS.dataset;OUT.mkdir(parents=True,exist_ok=False)
SOURCES={str(p.relative_to(ROOT)).replace(os.sep,'/'):driver.sha(p) for p in ROOT.rglob('*.py')}
CRITICAL=['core/model.py','scripts/exp2/exp2_run.py','scripts/exp3/exp3_governance_tasv2.py','state_digest.py']
OLD=Path('/home/njit/round2_final_20261028/results/c2')
started=time.monotonic();completed=[]
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
gov.full_trust=functools.partial(learning.compute_tasl_final_weights,gate_current=True)
save=driver.save
stamp=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
save(OUT/'run_manifest.json',dict(started_utc=stamp(),configuration=CONFIG,source_sha256=SOURCES,
 environment=dict(host=os.uname().nodename,python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),CUDA_VISIBLE_DEVICES=os.getenv('CUDA_VISIBLE_DEVICES'))))

def data(dataset,smoke=False):
 gov.set_seed(42)
 clients,test=gov.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=42,dataset=dataset)
 inventory=[dict(client=i,samples=len(l.dataset),labels_sha256=hashlib.sha256(l.dataset.labels.cpu().numpy().tobytes()).hexdigest(),data_sha256=hashlib.sha256(l.dataset.data.cpu().numpy().tobytes()).hexdigest()) for i,l in enumerate(clients)]
 assert sum(x['samples'] for x in inventory)==60000 and len(test.dataset)==10000
 if smoke:
  clients=[DataLoader(Subset(l.dataset,range(min(32,len(l.dataset)))),batch_size=16,shuffle=True) for l in clients]
  test=DataLoader(Subset(test.dataset,range(64)),batch_size=32,shuffle=False)
 else:
  save(OUT/(dataset+'_partition_inventory.json'),inventory)
  labels=datasets.MNIST(str(ROOT/'data/temp'),train=True,download=False).targets.numpy() if dataset=='mnist' else datasets.FashionMNIST(str(ROOT/'data/temp'),train=True,download=False).targets.numpy()
  indices=learning.dirichlet_partition(labels,10,.3,42)
  save(OUT/(dataset+'_partition_indices.json'),{str(c):list(map(int,ids)) for c,ids in indices.items()})
 return clients,test,inventory

def summarize(name,rows,elapsed,reused=False):
 attacked=[r for r in rows if r['attack_active']];honest=[r for r in rows if not r['attack_active']]
 rejected=lambda r:any(r.get(k,False) for k in ['v1_trust','v1_agg','v2_trust','v2_agg'])
 tp=sum(rejected(r) for r in attacked);fp=sum(rejected(r) for r in honest)
 valid=0;norm_leaks=0
 for row in rows:
  w={int(k):float(v) for k,v in json.loads(row['final_weights']).items()}
  pub={int(k):float(v) for k,v in json.loads(row['published_weights']).items()}
  norm={int(k):float(v) for k,v in json.loads(row['update_norms']).items()}
  weight_err=max([abs(pub.get(c,0)-v) for c,v in w.items()]+[0])
  if row['attack_active'] and (weight_err>1e-6 or row['proposal_relative_error']>1e-6):valid+=1
  median=float(np.median(list(norm.values())))
  leak=sum(w.get(c,0) for c,n in norm.items() if n>2.5*median)
  if leak>0:norm_leaks+=1
 assert norm_leaks==0,'Post-EMA norm exclusion failed'
 for row in rows:
  assert not json.loads(row['excluded_clients']),'Fixed participation must not permanently exclude any client'
  active=set(map(int,json.loads(row['update_norms'])))
  assert active==set(range(10)),'Every client must submit in every round'
  if row['attack_active']:
   assert {0,1,2,3}.issubset(set(json.loads(row['attack_targets']))),'Declared Byzantine targets no longer present; A3 may additionally suppress two honest clients'
   assert {0,1,2,3}.issubset(active)
 return dict(name=name,rounds=len(rows),avg_accuracy=float(np.mean([r['accuracy'] for r in rows])),
 best_accuracy=max(r['accuracy'] for r in rows),seconds=elapsed,reused=reused,
 honest_proposals=len(honest),attacked_proposals=len(attacked),meaningful_attacked_proposals=valid,
 true_rejections=tp,false_rejections=fp,false_acceptances=len(attacked)-tp,
 honest_acceptance=(len(honest)-fp)/len(honest) if honest else None,
 false_rejection_rate=fp/len(honest) if honest else None,precision=tp/(tp+fp) if tp+fp else None,
 recall=tp/len(attacked) if attacked else None,skipped=sum(r.get('skipped',0) for r in rows),
 norm_leak_rounds=norm_leaks,attack_types=dict(collections.Counter(r['attack_type'] for r in attacked)),
 interpretation='Interpret alongside paired complete-state and meaningful-tampering checks; process completion is not a scientific conclusion.')

def run(dataset,attack,name,group,clients,test,inventory,rounds=50,smoke=False):
 dest=OUT/('smoke' if smoke else 'matrix')/dataset/attack/name;dest.mkdir(parents=True,exist_ok=False)
 rows=None
 if rows is None:
  save(dest/'manifest.json',dict(started_utc=stamp(),dataset=dataset,client_attack=attack,group=group,rounds=rounds,configuration=CONFIG,partition=inventory,source_sha256=SOURCES,smoke=smoke,executor_schedule={r:vars(a) for r,a in driver.schedule(rounds).items()}))
  t=time.monotonic()
  result=gov.run_experiment_group(group,clients,test,rounds,driver.schedule(rounds),torch.device('cuda'),42,
    local_epochs=1 if smoke else 2,lr=.02,momentum=0.,weight_decay=0.,byzantine_clients=[0,1,2,3],
    ema_alpha=.7,trust_power=5.,max_weight_ratio=1.5,enable_exclusion=CONFIG['exclusion_enabled'],exclusion_budget=CONFIG['exclusion_cap'],
    client_attack=attack,client_attack_start=1,client_scaling_factor=5.,
    round_record_dir=str(dest/'rounds'),final_state_path=None if smoke else str(dest/'final_state.pt'))
  rows=result.round_logs;save(dest/'round_metrics.json',rows)
  summary=summarize(name,rows,time.monotonic()-t);save(dest/'summary.json',summary)
 print(json.dumps(dict(condition=str(dest.relative_to(OUT)),summary=json.loads((dest/'summary.json').read_text()))),flush=True)
 return rows

def pair(dataset,attack,left,right,smoke=False):
 matches=[a['state_hash']==b['state_hash'] for a,b in zip(left,right)]
 report=dict(dataset=dataset,attack=attack,rounds=len(left),complete_state_matches=sum(matches),mismatch_rounds=[i+1 for i,m in enumerate(matches) if not m],accuracy_matches=all(a['accuracy']==b['accuracy'] for a,b in zip(left,right)))
 save(OUT/('smoke' if smoke else 'matrix')/dataset/attack/'paired_consistency.json',report)
 assert len(left)==len(right) and all(matches),'Full TAS recovery/parent-state consistency failed; stop matrix'
 assert report['accuracy_matches']
 return report

def learning_check(dataset,attack,clients,test,reference):
 cfg=dict(experiment=dict(seed=42),training=dict(local_epochs=2,lr=.02,momentum=0.,weight_decay=0.),model=dict(num_classes=10),algorithm_params=dict(tasl=dict(trust_power=5.,weight_cap_ratio=1.5,ema_alpha=.7,norm_penalty=.8,fusion='multiplicative',exclusion=CONFIG['exclusion_enabled'],screen_tau=.6),scaling=dict(scaling_factor=5.)))
 t=time.monotonic();r=learning.run_single('tasl_full',attack,clients,test,cfg,gov.FedAvgCNN,10,4,50,torch.device('cuda'))
 rows=r['round_metrics'];save(OUT/'matrix'/dataset/attack/'learning_entry_rounds.json',rows)
 matches=[a['state_hash']==b['state_hash'] for a,b in zip(reference,rows)]
 report=dict(rounds=len(rows),complete_state_matches=sum(matches),seconds=time.monotonic()-t)
 save(OUT/'matrix'/dataset/attack/'learning_entry_check.json',report)
 assert len(matches)==50 and all(matches),'Independent learning entry differs from the governance reference'
 return report

try:
 for test in ['test_comment1_policy.py','test_final_protocol.py']:
  subprocess.run([sys.executable,test],cwd=ROOT,check=True,timeout=120)
 for dataset in [ARGS.dataset]:
  clients,test,inv=data(dataset,True)
  ref=run(dataset,'label_flip','reference','noexecutor',clients,test,inv,3,True)
  full=run(dataset,'label_flip','full_tas','trust_v1v2',clients,test,inv,3,True)
  pair(dataset,'label_flip',ref,full,True)
 save(OUT/'preflight.json',dict(passed=True,scope='CPU policy/role regressions and two-dataset three-round recovery pairing; smoke is not paper evidence'))
 for dataset in [ARGS.dataset]:
  for attack in CONFIG['client_attacks']:
   clients,test,inv=data(dataset)
   reference=run(dataset,attack,'reference','noexecutor',clients,test,inv)
   full=run(dataset,attack,'full_tas','trust_v1v2',clients,test,inv)
   paired=pair(dataset,attack,reference,full)
   learned=learning_check(dataset,attack,clients,test,reference)
   if dataset=='fashionmnist' and attack=='scaling':save(ROOT/'pilot_pair_passed.json',dict(paired=paired,learning_entry=learned,exclusion=False,all_clients_active=True,scope='Protocol-validity gate only; accuracy rankings do not control continuation'))
   for name,group in [('trust_only','trust_only'),('vs_only','trust_v1'),('vp_only','trust_v2')]:run(dataset,attack,name,group,clients,test,inv)
   completed.append(dict(dataset=dataset,attack=attack,pair=paired,learning_entry=learned))
   save(OUT/'progress.json',dict(state='running',finished_scenarios=completed,logical_conditions=len(completed)*5,updated_utc=stamp()))
 after={str(p.relative_to(ROOT)).replace(os.sep,'/'):driver.sha(p) for p in ROOT.rglob('*.py')}
 assert after==SOURCES,'Source changed during experiment'
 save(OUT/'completed.json',dict(state='completed',completed_scenarios=completed,logical_conditions=10,source_unchanged=True,seconds=time.monotonic()-started,finished_utc=stamp()))
except BaseException as error:
 save(OUT/'failure.json',dict(state='failed',error=repr(error),completed_scenarios=completed,seconds=time.monotonic()-started,utc=stamp()))
 raise
