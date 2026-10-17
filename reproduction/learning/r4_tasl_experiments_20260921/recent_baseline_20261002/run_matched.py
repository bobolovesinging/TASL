"""R4.8 matched aggregation comparison, seed 42, unchanged exp2 training loop."""
import argparse, csv, hashlib, importlib.util, json, os, platform, subprocess
from pathlib import Path
import numpy as np
import torch
import yaml
from lasa_adapter import lasa_update, UPSTREAM_COMMIT

ROOT = Path(__file__).resolve().parent
ap = argparse.ArgumentParser(); ap.add_argument('--attack',choices=['none','label_flip'],required=True)
ap.add_argument('--rounds',type=int,default=50);ap.add_argument('--output',required=True)
args = ap.parse_args()
source = ROOT/'pipeline/scripts/exp2/exp2_run.py'
expected = '8f4c447acab1f78e750592825643954dcf9ed8ba7402c907ca2cafe272a2ef83'
assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
spec = importlib.util.spec_from_file_location('frozen_learning',source)
pipe = importlib.util.module_from_spec(spec); spec.loader.exec_module(pipe)
cfg = yaml.safe_load((ROOT/'seed42.yaml').read_text())
cfg['training']['rounds'] = args.rounds
output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
device=torch.device('cuda');torch.set_num_threads(4)
pipe.set_seed(42)
loaders,test=pipe.load_fashionmnist_data(cfg)
from torchvision.datasets import FashionMNIST
labels=FashionMNIST(root=ROOT/'pipeline/data/temp',train=True,download=False).targets.numpy()
partition=pipe.dirichlet_partition(labels,10,alpha=0.3,seed=42)
for cid, loader in enumerate(loaders):
    assert np.array_equal(loader.dataset.labels,labels[partition[cid]])
(output/'partition.json').write_text(json.dumps(partition))
model_fn=lambda:pipe.create_model(cfg)
pipe.set_seed(42)
model=model_fn()
h=hashlib.sha256()
for k,v in model.state_dict().items():h.update(k.encode());h.update(v.numpy().tobytes())
env={'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
     'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
     'pipeline_sha256':expected,'LASA_author_commit':UPSTREAM_COMMIT,
     'LASA_defaults':{'sparsity':0.3,'lambda_n':1.0,'lambda_s':1.0},
     'initial_model_sha256':h.hexdigest(), 'seed':42,'attack':args.attack,
     'n_byzantine':0 if args.attack=='none' else 4,
     'aggregation_only_adaptation':True,'config':cfg,
     'individual_LASA_weights_available':False}
(output/'environment.json').write_text(json.dumps(env,indent=2))
(output/'pip_freeze.txt').write_text(subprocess.check_output([os.sys.executable,'-m','pip','freeze'],text=True))
rows=[]
original=pipe.fedavg_aggregate
for name in ('fedavg','tasl_full','lasa'):
    pipe.fedavg_aggregate = lasa_update if name=='lasa' else original
    result=pipe.run_single('fedavg' if name=='lasa' else name,args.attack,loaders,test,cfg,
                          model_fn,10,n_byz=0 if args.attack=='none' else 4,
                          rounds=args.rounds,device=device)
    # LASA enters the equal-weight dispatch only to reuse training; its actual
    # layer-wise selected weights are not the generic diagnostics from exp2.
    if name=='lasa':
        for row in result['round_metrics']:
            for key in ('accepted_total','honest_rejected','byzantine_nonzero','byzantine_weight_sum','weights_json'):
                row[key]=None
    (output/f'{name}.json').write_text(json.dumps(result,indent=2))
    rows.append({'algorithm':name,'attack':args.attack,'avg_acc':result['avg_acc'],
                 'best_acc':result['best_acc'],'avg_asr':result['avg_asr']})
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('COMPLETED',name,json.dumps(rows[-1]),flush=True)
pipe.fedavg_aggregate=original
(output/'complete.json').write_text(json.dumps({'trajectories':3,'round_records':args.rounds*3,
                                              'comment':'R4.8','completed':True},indent=2))
