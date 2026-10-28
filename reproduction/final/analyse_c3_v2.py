"""Client/class diagnostics and fixed-policy cap mismatch; no additional training."""
from pathlib import Path
import hashlib,json,sys,numpy as np,torch
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp3 import exp3_governance_tasv2 as m
OUT=ROOT/'results/c3_analysis_v2';OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2)
m.set_seed(42)
torch.use_deterministic_algorithms(True)
device=torch.device('cuda')
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda v:v.item())+'\n')
def ranks(x):
 order=np.argsort(x,kind='stable');result=np.empty(len(x));i=0
 while i<len(x):
  j=i+1
  while j<len(x) and x[order[j]]==x[order[i]]:j+=1
  result[order[i:j]]=(i+j-1)/2;i=j
 return result
def correlation(x,y):
 a,b=ranks(np.asarray(x)),ranks(np.asarray(y))
 return None if np.std(a)==0 or np.std(b)==0 else float(np.corrcoef(a,b)[0,1])
partition=json.loads((ROOT/'results/c2/partition_indices.json').read_text())
counts=np.asarray([partition['class_counts'][str(c)] for c in range(10)])
_,test=m.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=42,dataset='fashionmnist')
labels=test.dataset.labels.numpy();test_indices={c:[] for c in range(10)};rng=np.random.RandomState(20261028)
for y in range(10):
 ids=np.where(labels==y)[0];rng.shuffle(ids);prop=counts[:,y]/counts[:,y].sum();alloc=(prop*len(ids)).astype(int)
 for j in range(len(ids)-int(alloc.sum())):alloc[j%10]+=1
 offset=0
 for c,n in enumerate(alloc):test_indices[c].extend(ids[offset:offset+n].tolist());offset+=n
flat=[i for ids in test_indices.values() for i in ids];assert len(flat)==len(set(flat))==10000
save(OUT/'client_associated_test_indices.json',dict(seed=20261028,indices=test_indices,
 construction='Disjoint subsets of the untouched global test set allocated class by class in proportion to each training client class count; descriptive client-associated holdouts, not naturally collected per-client test data.'))
class_names=FashionMNIST(str(ROOT/'data/temp'),train=False,download=False).classes
summaries=[];client_records={};class_records={}
for attack in ['none','label_flip']:
 for cap in [0,2,4]:
  tag=f'{attack}_budget{cap}'
  if attack=='label_flip' and cap==4:
   rows=json.loads((ROOT/'results/c2/reference/round_metrics.json').read_text());checkpoint=ROOT/'results/c2/reference_final.pt'
  else:
   folder=ROOT/'results/c3_budget'/tag
   rows=json.loads((folder/'round_metrics.json').read_text());checkpoint=ROOT/'results/c3_budget'/(tag+'_final.pt')
  model=m.FedAvgCNN().to(device);model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model']);model.eval()
  predictions=[]
  with torch.no_grad():
   for x,y in test:predictions.append(model(x.to(device)).argmax(1).cpu().numpy())
  predictions=np.concatenate(predictions);accuracy=float(np.mean(predictions==labels)*100)
  assert abs(accuracy-rows[-1]['accuracy'])<1e-10,(tag,accuracy,rows[-1]['accuracy'])
  actual=set() if attack=='none' else set(range(4));honest=set(range(10))-actual
  weights=[{int(c):w for c,w in json.loads(row['final_weights']).items()} for row in rows]
  exclusions=[set(json.loads(row['excluded_clients'])) for row in rows]
  clients=[]
  for c in range(10):
   n=int(counts[c].sum());profile=counts[c]/n;ids=test_indices[c]
   zero=sum(w.get(c,0.)<=1e-12 for w in weights)
   first=next((r['round'] for r,e in zip(rows,exclusions) if c in e),None)
   clients.append(dict(client=c,training_samples=n,dominant_class=class_names[int(np.argmax(profile))],
      label_total_variation=float(.5*np.abs(profile-.1).sum()),zero_weight_rounds=zero,opportunities=50,
      excluded_at_round=first,associated_test_samples=len(ids),associated_test_accuracy=float(np.mean(predictions[ids]==labels[ids])*100),honest=c in honest))
  client_records[tag]=clients
  classes=[dict(class_id=y,class_name=class_names[y],test_samples=int(np.sum(labels==y)),accuracy=float(np.mean(predictions[labels==y]==y)*100)) for y in range(10)]
  class_records[tag]=classes
  summaries.append(dict(condition='Clean' if attack=='none' else '40% LF',budget=cap,actual_faults=len(actual),
    avg_accuracy=float(np.mean([r['accuracy'] for r in rows])),final_accuracy=accuracy,
    final_lf_asr=float(np.mean(predictions==9-labels)*100) if actual else None,
    honest_zero_opportunities=sum(sum(w.get(c,0.)<=1e-12 for c in honest) for w in weights),honest_opportunities=50*len(honest),
    honest_excluded=len(exclusions[-1]&honest),byzantine_excluded=len(exclusions[-1]&actual),
    correlation_training_size_zero_rate=correlation([c['training_samples'] for c in clients if c['honest']],[c['zero_weight_rounds'] for c in clients if c['honest']]),
    correlation_label_TV_zero_rate=correlation([c['label_total_variation'] for c in clients if c['honest']],[c['zero_weight_rounds'] for c in clients if c['honest']]),
    final_model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()))
save(OUT/'budget_summary.json',summaries);save(OUT/'client_metrics.json',client_records);save(OUT/'class_metrics.json',class_records)
save(OUT/'manifest.json',dict(comment='R4 round2 Comment 3',seed=42,inference='Same deterministic CUDA backend/batch size as recorded training evaluation; CPU inference diagnostic differed at a class-decision boundary.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    training_partition_sha256=hashlib.sha256((ROOT/'results/c2/partition_indices.json').read_bytes()).hexdigest(),
    scope='Descriptive per-class and client-associated holdout accuracy from final global models, zero-weight opportunities including post-exclusion rounds, fixed rule with cap 0/2/4. One partition; no causal fairness or subgroup-generalisation claim.'))
print(json.dumps(summaries),flush=True)
