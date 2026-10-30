"""Regenerate the two final V_P trajectories for one dataset in a fresh workspace."""
from pathlib import Path
import argparse,json,os,subprocess,sys
BASE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['mnist','fashionmnist'],required=True);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
w=a.workspace.resolve();repo=BASE.parent.parent
if w==repo or repo in w.parents:p.error('Use a fresh workspace outside the repository')
files=json.loads((BASE/'records/normalized_sources.json').read_text(encoding='utf-8'))['vp_only']
w.mkdir(parents=True,exist_ok=False)
for name,text in files.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 dest=w/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(text.encode('utf-8'))
# The recorded run reused one already-finished row; reproduction regenerates it as well.
script=w/'run_matrix_normalized.py';s=script.read_text(encoding='utf-8');old="  if ARGS.dataset=='fashionmnist' and attack=='scaling':continue"
assert old in s;s=s.replace(old,'  # Regenerate both attacks; no previously completed row is reused.')
assert '/home/njit/' not in s;script.write_text(s,encoding='utf-8')
env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4')
cmd=[sys.executable,str(script),'--dataset',a.dataset];print('Command:',cmd)
if not a.prepare_only:subprocess.run(cmd,cwd=w,env=env,check=True)
