"""Run each matched condition on one GPU with the agreed thermal stop rule."""
import argparse,csv,datetime,json,os,signal,subprocess,sys,time
from pathlib import Path
ap=argparse.ArgumentParser();ap.add_argument('--gpu',type=int,required=True)
ap.add_argument('--attack',choices=['none','label_flip','exposed','anchor_norm'],required=True);a=ap.parse_args()
root=Path(__file__).resolve().parent
status={'comment':'R4.6' if a.attack in ('exposed','anchor_norm') else 'R4.8','state':'running','gpu':a.gpu,'attack':a.attack,'maximum_core_temperature_C':0,
        'thermal_rule':'>=86 C immediately or >=82 C for three consecutive five-second samples'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
status['started_utc']=now()
def save():(root/f'{a.attack}_status.json').write_text(json.dumps(status,indent=2))
save();p=None
try:
 with (root/f'{a.attack}_telemetry.csv').open('w',newline='') as f, (root/f'{a.attack}.log').open('w') as log:
  w=csv.writer(f);w.writerow(['utc','gpu','core_temp_C','power_W','fan_pct','util_pct','memory_MiB']);high=0
  while True:
   raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,power.draw,fan.speed,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
   row=next([s.strip() for s in r.split(',')] for r in raw.splitlines() if int(r.split(',')[0])==a.gpu)
   temp=int(row[1]);w.writerow([now()]+row);f.flush()
   status['maximum_core_temperature_C']=max(status['maximum_core_temperature_C'],temp)
   high=high+1 if temp>=82 else 0
   if temp>=86 or high>=3:raise RuntimeError('Thermal threshold reached')
   if p is None:
    if temp>=75:raise RuntimeError('GPU too warm for launch')
    cmd=([sys.executable,'-u',str(root/'run_trust_sweep.py'),'--group',a.attack] if a.attack in ('exposed','anchor_norm') else
         [sys.executable,'-u',str(root/'run_matched.py'),'--attack',a.attack,'--output',str(root/'results'/a.attack)])
    status['command']=cmd
    p=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(a.gpu),OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1'),cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status['pid']=p.pid;save()
   rc=p.poll()
   if rc is not None:
    status.update(exit_code=rc,state='completed' if rc==0 else 'failed');break
   time.sleep(5)
except Exception as e:
 status.update(state='stopped',reason=str(e))
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGINT)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
finally:status['finished_utc']=now();save();print(json.dumps(status),flush=True)
