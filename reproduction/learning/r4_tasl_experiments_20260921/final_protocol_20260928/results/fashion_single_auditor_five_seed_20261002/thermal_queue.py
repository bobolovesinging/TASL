"""Supervise one existing-protocol queue per physical GPU; no algorithm changes."""
import argparse, csv, datetime, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--attack',choices=['lf','scaling'],required=True)
parser.add_argument('--gpu',type=int,required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
source_root = Path('/home/njit/tasl_r4_final_protocol_20260928')
source = source_root/'scripts/exp3/exp3_governance_tasv2.py'
expected = 'f71067a01c7af357c664306aed944a64c440e53df537a0f2f7e8f0311311e987'
assert hashlib.sha256(source.read_bytes()).hexdigest()==expected
env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu),OMP_NUM_THREADS='4')
status_path = root/f'{args.attack}_status.json'
def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(): status_path.write_text(json.dumps(status,indent=2))
status = {'state':'running','started_utc':now(),'physical_gpu':args.gpu,
 'comments':['R4.9','R2.2','R2.4'],'source_sha256':expected,'completed':[],
 'environment':{'python':sys.executable,'CUDA_VISIBLE_DEVICES':env['CUDA_VISIBLE_DEVICES'],'OMP_NUM_THREADS':env['OMP_NUM_THREADS']},
 'thermal_policy':'Stop at >=86 C immediately or >=82 C for three consecutive 5-second samples; core temperature only.',
 'maximum_core_temperature_C':0}
save()
p=None
try:
 with (root/f'{args.attack}_gpu_telemetry.csv').open('w',newline='') as telemetry:
  writer=csv.writer(telemetry);writer.writerow(['utc','seed','gpu_index','core_temp_C','power_W','fan_pct','gpu_util_pct','memory_MiB'])
  for seed in (42,123,2024,3407,7711):
   output=root/'results'/f'{args.attack}_seed{seed}'
   if output.exists(): raise RuntimeError(f'Refusing to overwrite existing run: {output}')
   config=root/'configs'/f'fashion_{args.attack}_seed{seed}.yaml'
   command=[sys.executable,'-u',str(source),'--config',str(config)]
   status.update(current_seed=seed,current_command=command);save()
   high=0
   with (root/f'{args.attack}_seed{seed}.log').open('w') as log:
    p=None
    while True:
     raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,power.draw,fan.speed,utilization.gpu,memory.used','--format=csv,noheader,nounits'],text=True,timeout=10)
     gpu=next([s.strip() for s in row.split(',')] for row in raw.strip().splitlines() if int(row.split(',')[0])==args.gpu)
     temp=int(gpu[1]);writer.writerow([now(),seed]+gpu);telemetry.flush()
     status['maximum_core_temperature_C']=max(status['maximum_core_temperature_C'],temp)
     high=high+1 if temp>=82 else 0
     if temp>=86 or high>=3: raise RuntimeError('GPU core temperature stop threshold reached')
     if p is None:
      if temp>=75: raise RuntimeError('GPU too warm for launch')
      p=subprocess.Popen(command,env=env,cwd=source_root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
      status['current_pid']=p.pid;save()
     result=p.poll()
     if result is not None:
      if result!=0: raise RuntimeError(f'seed {seed} exited with {result}')
      status['completed'].append({'seed':seed,'finished_utc':now(),'exit_code':result});save();break
     time.sleep(5)
  status['state']='completed'
except Exception as e:
 status.update(state='stopped',reason=str(e))
 if p is not None and p.poll() is None:
  os.killpg(p.pid,signal.SIGINT)
  try:p.wait(timeout=15)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=10)
finally:
 status['finished_utc']=now();save();print(json.dumps(status),flush=True)
