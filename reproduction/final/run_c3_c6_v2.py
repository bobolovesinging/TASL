"""Matched final-method repetitions and a focused fixed-policy budget study."""
from pathlib import Path
import argparse,copy,hashlib,json,random,sys,time,shutil
import numpy as np,torch
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov
ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True,choices=[0,1]);args=ap.parse_args()
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
def finite(value):
 if isinstance(value,dict):return {k:finite(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [finite(v) for v in value]
 if isinstance(value,(float,np.floating)) and not np.isfinite(value):return None
 return value
def save(path,value):driver.save(path,finite(value))
def existing(dest):
 if (dest/'summary.json').exists() and (dest/'metrics.json').exists() and (dest/'final_model.pt').exists():
  manifest=json.loads((dest/'manifest.json').read_text())
  assert manifest['source_sha256']['scripts/exp2/exp2_run.py']==driver.sha(ROOT/'scripts/exp2/exp2_run.py')
  return json.loads((dest/'summary.json').read_text())
 if dest.exists():
  failed=ROOT/'failed_attempts'/dest.name
  assert ROOT.resolve() in dest.resolve().parents and ROOT.resolve() in failed.resolve().parents
  failed.parent.mkdir(exist_ok=True)
  if failed.exists():raise FileExistsError(failed)
  shutil.move(str(dest),str(failed))
 return None
def config(seed):
 return dict(experiment=dict(seed=seed),data=dict(dataset='fashionmnist',batch_size=128,n_clients=10,alpha=.3),
  training=dict(local_epochs=2,lr=.02,momentum=0.,weight_decay=0.),model=dict(num_classes=10),
  algorithm_params=dict(bsp=dict(enabled=False),tasl=dict(trust_power=5.,weight_cap_ratio=1.5,
  ema_alpha=.7,norm_penalty=.8,fusion='multiplicative',exclusion=True,screen_tau=.6)))

if args.shard==0 and not (OUT/'c3_budget/completed.json').exists():
 budget=OUT/'c3_budget';budget.mkdir(exist_ok=True)
 save(budget/'study_manifest.json',dict(comment='R4 round2 Comment 3',actual_faults=[0,4],assumed_exclusion_budgets=[0,2,4],
  fixed='All training/trust constants and progressive two-violation exclusion rule fixed; only exclusion cap changed.',
  reused='40% LF cap4 reference/final checkpoint from C2',seed=42))
 for attack,b in [('none',0),('none',2),('none',4),('label_flip',0),('label_flip',2)]:
  name=f'{attack}_budget{b}'
  if (budget/name/'summary.json').exists():continue
  if (budget/name).exists():
   failed=ROOT/'failed_attempts'/name;failed.parent.mkdir(exist_ok=True)
   assert ROOT.resolve() in failed.resolve().parents
   shutil.move(str(budget/name),str(failed))
  driver.run_one(budget,name,True,'noexecutor',attack,50,
      extra=dict(exclusion_budget=b,final_state_path=str(budget/(name+'_final.pt'))))
 save(budget/'completed.json',dict(comment='R4 round2 Comment 3',new_runs=5,reused_runs=1))

import lasa_adapter
original_aggregate=learning.fedavg_aggregate
jobs=[(seed,attack,method) for seed in [42,123,2024,3407,7711]
      for attack in ['none','label_flip'] for method in ['fedavg','fltrust','tasl_full','lasa']]
random.Random(20261028).shuffle(jobs)
done=[]
for index,(seed,attack,method) in enumerate(jobs):
 if index%2!=args.shard:continue
 dest=OUT/'c6_repeated'/f'{method}_{attack}_seed{seed}'
 prior=existing(dest)
 if prior is not None:
  done.append(prior);continue
 dest.mkdir(parents=True,exist_ok=False)
 # The unchanged LASA CUDA median lacks a Torch deterministic-kernel guarantee.
 # Retain author computation; record that warning rather than substitute a method.
 torch.use_deterministic_algorithms(True,warn_only=(method=='lasa'))
 cfg=config(seed);gov.set_seed(seed)
 clients,test=gov.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=seed,dataset='fashionmnist')
 proxy=learning.create_proxy_loader('fashionmnist',n_per_class=50,batch_size=64) if method=='fltrust' else None
 tracked=[]
 def factory():
  model=gov.FedAvgCNN()
  if not tracked:tracked.append(model)
  return model
 manifest=dict(comments=['R4 round2 Comment 6','R4 round2 Comment 2 final learning'],method=method,attack=attack,
   configuration=cfg,actual_byzantine_clients=[] if attack=='none' else list(range(4)),
   exclusion_budget=4,source_sha256={str(p.relative_to(ROOT)):driver.sha(p) for p in [Path(__file__),ROOT/'scripts/exp2/exp2_run.py',ROOT/'core/model.py',ROOT/'lasa_adapter.py']},
   partition=[dict(client=c,samples=len(l.dataset),labels_sha256=hashlib.sha256(l.dataset.labels.numpy().tobytes()).hexdigest(),data_sha256=hashlib.sha256(l.dataset.data.numpy().tobytes()).hexdigest()) for c,l in enumerate(clients)],
   lasa_upstream=lasa_adapter.UPSTREAM_COMMIT if method=='lasa' else None,
   root_samples=500 if method=='fltrust' else 0,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
   deterministic_kernels='warn_only_for_author_LASA_CUDA_median' if method=='lasa' else 'required')
 save(dest/'manifest.json',manifest)
 learning.fedavg_aggregate=lasa_adapter.lasa_update if method=='lasa' else original_aggregate
 start=time.monotonic()
 try:
  result=learning.run_single('fedavg' if method=='lasa' else method,attack,clients,test,cfg,factory,10,4,50,torch.device('cuda'),proxy_loader=proxy)
 finally:learning.fedavg_aggregate=original_aggregate
 checkpoint={k:v.detach().cpu().clone() for k,v in tracked[0].state_dict().items()}
 torch.save(checkpoint,dest/'final_model.pt')
 save(dest/'metrics.json',result)
 if method=='tasl_full' and attack=='label_flip' and seed==42:
  ref=json.loads((OUT/'c2/reference/round_metrics.json').read_text())
  assert all(a['state_hash']==b['state_hash'] for a,b in zip(ref,result['round_metrics']))
 summary=dict(method=method,attack=attack,seed=seed,avg=result['avg_acc'],final=result['last_acc'],lf_asr=result['avg_asr'] if attack=='label_flip' else None,seconds=time.monotonic()-start)
 save(dest/'summary.json',summary);done.append(summary);print(json.dumps(summary),flush=True)
save(OUT/f'c6_shard{args.shard}_completed.json',dict(comment='R4 round2 Comment 6',shard=args.shard,runs=done))
