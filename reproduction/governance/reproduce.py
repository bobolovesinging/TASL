from pathlib import Path
import argparse,hashlib,os,shutil,subprocess,sys
BASE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--workspace',type=Path,required=True);ap.add_argument('--budget',type=int,choices=[0,4],required=True);ap.add_argument('--dataset',choices=['mnist','fashionmnist']);a=ap.parse_args()
if a.budget==0 and not a.dataset:ap.error('--dataset is required with budget zero')
if a.budget==4 and a.dataset:ap.error('budget four evaluates both datasets; omit --dataset')
source=BASE/'code'/('cap'+str(a.budget));dest=a.workspace.resolve()
if dest==BASE or BASE in dest.parents:raise ValueError('Choose a workspace outside the downloaded release')
dest.mkdir(parents=True,exist_ok=True)
for p in source.rglob('*'):
 if p.is_file():
  out=dest/p.relative_to(source);out.parent.mkdir(parents=True,exist_ok=True)
  if out.exists():
   if hashlib.sha256(out.read_bytes()).digest()!=hashlib.sha256(p.read_bytes()).digest():raise ValueError('Workspace source differs: '+str(out))
  else:shutil.copy2(p,out)
command=[sys.executable,'-u','run_matrix.py']+(['--dataset',a.dataset] if a.budget==0 else [])
raise SystemExit(subprocess.run(command,cwd=dest).returncode)
