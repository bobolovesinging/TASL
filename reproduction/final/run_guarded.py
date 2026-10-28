"""Bounded single-GPU launcher; thermal policy matches the previous experiments."""
import argparse,csv,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--script',required=True)
args=ap.parse_args()
assert args.tag.replace('_','').isalnum()
script=(ROOT/args.script).resolve();assert script.parent==ROOT and script.suffix=='.py'
prefix=ROOT/('guard_'+args.tag)
state=dict(state='preflight',comment='R4 round2 C1',gpu=1,maximum_temperature_C=0)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save():prefix.with_suffix('.json').write_text(json.dumps(state,indent=2)+'\n')
state['started_utc']=stamp();save();p=None;start=time.monotonic()
try:
 with prefix.with_suffix('.log').open('x') as log,prefix.with_suffix('.csv').open('x',newline='') as fh:
  writer=csv.writer(fh);writer.writerow(['utc','temp_C','util_pct','memory_MiB']);high=0
  while True:
   raw=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=temperature.gpu,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
   temp,util,mem=[int(s.strip()) for s in raw.strip().split(',')]
   writer.writerow([stamp(),temp,util,mem]);fh.flush()
   state.update(current_temperature_C=temp,maximum_temperature_C=max(state['maximum_temperature_C'],temp),last_sample_utc=stamp())
   high=high+1 if temp>=82 else 0
   if temp>=86 or high>=3:raise RuntimeError('Thermal stop')
   if time.monotonic()-start>3600:raise RuntimeError('One-hour diagnostic/run budget reached')
   if p is None:
    if temp>=75 or mem>=1024:raise RuntimeError('GPU 1 unavailable at preflight')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONHASHSEED='42')
    p=subprocess.Popen([sys.executable,'-u',str(script)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(state='running',pid=p.pid,script=str(script))
   rc=p.poll();save()
   if rc is not None:
    state.update(state='completed' if rc==0 else 'failed',exit_code=rc);break
   time.sleep(5)
except BaseException as err:
 state.update(state='stopped',reason=str(err))
 if p and p.poll() is None:
  os.killpg(p.pid,signal.SIGINT)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
finally:state['finished_utc']=stamp();save();print(json.dumps(state),flush=True)
if state.get('state')!='completed':sys.exit(1)
