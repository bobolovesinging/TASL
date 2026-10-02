"""Capture actual environment and deterministic dataset partitions, without training."""
import hashlib, importlib.util, json, os, platform, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
source_root=Path('/home/njit/tasl_r4_final_protocol_20260928')
source=source_root/'scripts/exp3/exp3_governance_tasv2.py'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(source)=='f71067a01c7af357c664306aed944a64c440e53df537a0f2f7e8f0311311e987'
spec=importlib.util.spec_from_file_location('final_governance_source',source)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
import numpy, torch, torchvision, yaml
from scripts.exp2.exp2_run import dirichlet_partition
torch.set_num_threads(4)
env={'python':sys.version,'platform':platform.platform(),'hostname':platform.node(),
 'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,'yaml':yaml.__version__,
 'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
 'gpus':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version','--format=csv,noheader'],text=True),
 'scope':'Reproduction record captured during existing runs; does not establish public release or recover original submitted table sources.'}
(root/'environment.json').write_text(json.dumps(env,indent=2))
(root/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
files=[source]+list((source_root/'core').rglob('*.py'))+list((source_root/'scripts/exp2').glob('*.py'))
manifest={'source_files':[{ 'path':str(p.relative_to(source_root)),'sha256':sha(p)} for p in sorted(set(files))],
 'dataset_files':[{'path':str(p.relative_to(source_root)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted((source_root/'data/temp/FashionMNIST/raw').glob('*')) if p.is_file()],
 'partitions':[], 'partition_capture':'Rebuilt from unchanged source and training seed; not directly emitted by the training process.'}
dataset=torchvision.datasets.FashionMNIST(root=str(source_root/'data/temp'),train=True,download=False)
labels=dataset.targets.numpy()
for seed in (42,123,2024,3407,7711):
 loaders,test=m.get_data_loaders(client_num=10,batch_size=128,alpha=.3,seed=seed,dataset='fashionmnist')
 partitions=dirichlet_partition(labels,10,alpha=.3,seed=seed)
 entries=[]
 for cid,loader in loaders.items() if isinstance(loaders,dict) else enumerate(loaders):
  subset=loader.dataset
  indices=[int(x) for x in partitions[cid]]
  assert numpy.array_equal(labels[indices],subset.labels.numpy())
  assert numpy.array_equal(dataset.data.numpy()[indices].astype(numpy.float32)[:,None]/255.,subset.data.numpy())
  counts=numpy.bincount(labels[indices],minlength=10).tolist()
  entries.append({'client':int(cid),'size':len(indices),'label_counts':counts,'indices':indices})
 unique=[x for e in entries for x in e['indices']]
 assert len(unique)==len(set(unique)) and len(unique)==60000,(seed,len(unique))
 path=root/f'partition_seed{seed}.json'
 path.write_text(json.dumps({'seed':seed,'alpha':.3,'clients':entries},separators=(',',':')))
 manifest['partitions'].append({'seed':seed,'file':path.name,'sha256':sha(path),'client_sizes':[e['size'] for e in entries]})
(root/'reproduction_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps({'partitions':len(manifest['partitions']),'hashed_source_files':len(manifest['source_files']),'dataset_files':len(manifest['dataset_files'])}))
