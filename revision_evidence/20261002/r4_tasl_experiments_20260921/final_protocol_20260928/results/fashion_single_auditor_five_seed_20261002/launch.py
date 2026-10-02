import json, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
if any(root.glob('*_status.json')):
 raise RuntimeError('Queue already launched; inspect status rather than duplicate it.')
jobs=[]
for attack,gpu in [('lf',0),('scaling',1)]:
 p=subprocess.Popen([sys.executable,'-u',str(root/'thermal_queue.py'),'--attack',attack,'--gpu',str(gpu)],stdin=subprocess.DEVNULL,stdout=(root/f'{attack}_queue.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
 jobs.append({'attack':attack,'gpu':gpu,'supervisor_pid':p.pid})
(root/'launch.json').write_text(json.dumps(jobs,indent=2))
print(json.dumps(jobs))
