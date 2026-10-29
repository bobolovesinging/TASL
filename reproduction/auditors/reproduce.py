"""Adapt execution paths in a fresh workspace; keep released evidence immutable."""
from pathlib import Path
import argparse,os,shutil,subprocess,sys
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('suite',choices=['verify','train','benchmark','sampling']);p.add_argument('--dataset',choices=['mnist','fashionmnist']);p.add_argument('--workspace',type=Path);p.add_argument('--capture-workspace',type=Path);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
if a.suite=='verify':
 from verify import main
 main();raise SystemExit(0)
if a.workspace is None:p.error('--workspace is required')
w=a.workspace.resolve()
if w==BASE or BASE in w.parents or BASE.parent.parent==w or BASE.parent.parent in w.parents:p.error('Use a fresh workspace outside the repository')
w.mkdir(parents=True,exist_ok=False)
for src in (BASE/'executed_sources').rglob('*.py'):
 dest=w/src.relative_to(BASE/'executed_sources');dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42')
if a.suite=='train':
 if a.dataset is None:p.error('--dataset is required for training')
 for cap in [0,4]:
  for attack in ['label_flip','scaling']:
   relative=('cap0/'+a.dataset+'/matrix/'+a.dataset+'/'+attack if cap==0 else 'cap4/matrix/'+a.dataset+'/'+attack)
   for name in ['reference/round_metrics.json','vp_only/manifest.json']:
    src=BASE.parent/'governance/records'/relative/name;dest=w/'reference_records'/relative/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
 source=w/'run_vp_final.py';text=source.read_text(encoding='utf-8')
 text=text.replace("Path('/home/njit/round2_C2_fixed_v2_20261029/results')","(ROOT/'reference_records/cap0')").replace("Path('/home/njit/round2_C2_matrix_20261029/results/matrix')","(ROOT/'reference_records/cap4/matrix')")
 assert '/home/njit/' not in text
 source.write_text(text,encoding='utf-8');cmd=[sys.executable,str(source),'--dataset',a.dataset]
else:
 if a.capture_workspace is None:p.error('--capture-workspace is required for replay')
 captures=a.capture_workspace.resolve();assert (captures/'results/c2/captured/round_50.pt').is_file(),'Regenerate captures with the final/c2 suite first'
 env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='2')
 if a.suite=='benchmark':cmd=[sys.executable,str(w/'benchmark_final.py'),'--base',str(captures),'--candidate',str(w),'--output',str(w/'measurements')]
 else:
  source=w/'sampling_final.py';text=source.read_text(encoding='utf-8').replace("Path('/home/njit/tas_vp_final_20261029')","Path(__file__).resolve().parent").replace("Path('/home/njit/round2_final_20261028/results/c2/captured')",repr(str(captures/'results/c2/captured')))
  text=text.replace("base="+repr(str(captures/'results/c2/captured')),"base=Path("+repr(str(captures/'results/c2/captured'))+")")
  # The executed script writes to its original analysis folder; redirect only output paths.
  text=text.replace("Path('/home/njit/tas_vp_final_analysis_20261029/sampling_final.json')","Path(__file__).resolve().parent/'sampling_final.json'")
  source.write_text(text,encoding='utf-8');cmd=[sys.executable,str(source)]
print('Command:',cmd)
if not a.prepare_only:subprocess.run(cmd,cwd=w,env=env,check=True)
