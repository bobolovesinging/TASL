"""Prepare a fresh output workspace and run one explicitly selected final suite."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys,urllib.request
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('suite',choices=['verify','c1','c2','c3-c6','c3-analysis','c4','c7','feedback','hbs','plots'])
parser.add_argument('--workspace',type=Path);parser.add_argument('--shard',choices=['0','1'],default='0');args=parser.parse_args()
if args.suite=='verify':
 from verify_records import main
 main();raise SystemExit(0)
if args.workspace is None:parser.error('Training/replay requires an explicit fresh --workspace path; public records are read-only inputs.')
workspace=args.workspace.resolve();workspace.mkdir(parents=True,exist_ok=True)
if workspace==BASE or BASE in workspace.parents:parser.error('Use a workspace outside this release directory.')
for p in BASE.rglob('*.py'):
 if any(x in p.parts for x in ['records','__pycache__','LASA_upstream']):continue
 dest=workspace/p.relative_to(BASE);dest.parent.mkdir(parents=True,exist_ok=True)
 if dest.exists():
  if dest.read_bytes()!=p.read_bytes():raise RuntimeError(f'Different existing source: {dest}')
 else:shutil.copy2(p,dest)
if args.suite=='c3-c6':
 commit='8477367a4e8708cde264f7572805040c650af59f'
 expected={'algorithms/defense/lasa.py':'f92a6ff6b603b697bcf8631fada72e4365591705fc7376b6644dac05382dd1a2',
 'utils/mask_help.py':'e9e4591f480d8a3ca425903cfe65fbc29ed66f3f124c0162f2fcbc6bbb787521',
 'utils/__init__.py':'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'}
 for name,digest in expected.items():
  dest=workspace/'LASA_upstream'/name;dest.parent.mkdir(parents=True,exist_ok=True)
  if not dest.exists():dest.write_bytes(urllib.request.urlopen(f'https://raw.githubusercontent.com/JiiahaoXU/LASA/{commit}/{name}',timeout=60).read())
  if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise RuntimeError(f'Upstream mismatch: {name}')
commands={'c1':['run_c1_serialization_safe.py'],'c2':['run_c2.py'],'c3-c6':['run_c3_c6_v2.py','--shard='+args.shard],
 'c3-analysis':['analyse_c3_v2.py'],'c4':['benchmark_c4.py'],'c7':['replay_c7.py'],
 'feedback':['run_feedback_and_adaptive.py'],'hbs':['hbs_final_replay_v2.py'],'plots':['plot_learning_curves.py']}
subprocess.run([sys.executable]+commands[args.suite],cwd=workspace,check=True)
