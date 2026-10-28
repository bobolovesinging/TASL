"""Minimal final-implementation C2 comparison authorized by the author."""
from pathlib import Path
import json,hashlib,sys,time
import numpy as np,torch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
import run_experiments as driver
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as gov
OUT=ROOT/'results/c2';OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
for name,group,capture in [('reference','noexecutor',True),('trust_only','trust_only',False),('full_tas','trust_v1v2',False)]:
 extra=dict(exclusion_budget=4,final_state_path=str(OUT/(name+'_final.pt')))
 if capture:extra['capture_round_dir']=str(OUT/'captured')
 driver.run_one(OUT,name,True,group,'label_flip',50,extra=extra)
driver.compare(OUT,'reference','full_tas')
clients,test,partitions=driver.loaders()
reference=json.loads((OUT/'reference/round_metrics.json').read_text())
cfg=dict(experiment=dict(seed=42),training=dict(local_epochs=2,lr=.02,momentum=0.,weight_decay=0.),
 model=dict(num_classes=10),algorithm_params=dict(tasl=dict(trust_power=5.,weight_cap_ratio=1.5,
 ema_alpha=.7,norm_penalty=.8,fusion='multiplicative',exclusion=True,screen_tau=.6)))
learn=learning.run_single('tasl_full','label_flip',clients,test,cfg,gov.FedAvgCNN,10,4,50,torch.device('cuda'))
driver.save(OUT/'learning_entry_rounds.json',learn['round_metrics'])
matches=[a['state_hash']==b['state_hash'] for a,b in zip(reference,learn['round_metrics'])]
assert len(matches)==50 and all(matches),'Final learning/governance full-state mismatch'
from torchvision.datasets import FashionMNIST
data=FashionMNIST(str(ROOT/'data/temp'),train=True,download=False)
indices=learning.dirichlet_partition(data.targets.numpy(),10,.3,42)
driver.save(OUT/'partition_indices.json',dict(seed=42,alpha=.3,training_indices=indices,
 class_counts={c:np.bincount(data.targets.numpy()[ids],minlength=10).tolist() for c,ids in indices.items()}))
captures=[]
for p in sorted((OUT/'captured').glob('*.pt')):
 captures.append(dict(file=p.name,bytes=p.stat().st_size,sha256=driver.sha(p)))
driver.save(OUT/'captured_index.json',captures)
summary={name:json.loads((OUT/name/'summary.json').read_text()) for name in ['reference','trust_only','full_tas']}
driver.save(OUT/'completed.json',dict(comment='R4 second round Comment 2',conditions=summary,
 learning_governance_state_matches=sum(matches),reference_full_tas_state_matches=50,
 dimension=sum(p.numel() for p in gov.FedAvgCNN().parameters()),
 scope='Single-host integrated learning/verification/recovery, with two independently recomputed audit-role references. Historical Tables 4/5 are not re-labelled as these new runs.'))
print(json.dumps(dict(state='completed',comment='R4 second round Comment 2',conditions=summary,learning_governance_state_matches=sum(matches))),flush=True)
