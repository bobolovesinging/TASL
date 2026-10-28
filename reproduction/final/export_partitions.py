"""Record original training indices and verify against every final-run loader hash."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as learning
from scripts.exp3 import exp3_governance_tasv2 as m
from torchvision.datasets import FashionMNIST
labels=FashionMNIST(str(ROOT/'data/temp'),train=True,download=True).targets.numpy()
out=ROOT/'results/partitions';out.mkdir(exist_ok=True)
for seed in [42,123,2024,3407,7711]:
 indices=learning.dirichlet_partition(labels,10,.3,seed)
 flat=[i for x in indices.values() for i in x];assert len(flat)==60000 and len(set(flat))==60000
 clients,_,=m.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=seed,dataset='fashionmnist')
 recorded=json.loads((ROOT/'results/c6_repeated'/f'tasl_full_none_seed{seed}'/'manifest.json').read_text())['partition']
 for c,loader in enumerate(clients):
  assert np.array_equal(labels[indices[c]],loader.dataset.labels.numpy())
  assert hashlib.sha256(loader.dataset.data.numpy().tobytes()).hexdigest()==recorded[c]['data_sha256']
  assert hashlib.sha256(loader.dataset.labels.numpy().tobytes()).hexdigest()==recorded[c]['labels_sha256']
 (out/f'seed{seed}.json').write_text(json.dumps(dict(seed=seed,alpha=.3,clients=10,training_indices=indices,class_counts={c:np.bincount(labels[ix],minlength=10).tolist() for c,ix in indices.items()}))+'\n')
print('Verified all 300000 partition assignments against executed-run data hashes')
