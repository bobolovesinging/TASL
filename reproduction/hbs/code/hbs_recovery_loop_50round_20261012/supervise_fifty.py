"""Detached supervised paired capture, with thermal protection and queued replay."""
import os,json,signal,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def save(path,data):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)

def main():
    if (HERE/'job_status.json').exists():raise FileExistsError('Existing job status; refuse duplicate launch')
    (HERE/'supervisor.pid').write_text(str(os.getpid()))
    env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
             TASL_DATA_ROOT='/home/njit/tasl_r4_final_protocol_20260928/data/temp',
             CUBLAS_WORKSPACE_CONFIG=':4096:8')
    jobs={};streams=[];phase='capture';hot={};max_temps={}
    try:
        for group,device in (('noexecutor','cuda:0'),('trust_v1v2','cuda:1')):
            stream=(HERE/(group+'.log')).open('w');streams.append(stream)
            command=[sys.executable,'-u',str(HERE/'capture_fifty.py'),'--group',group,'--device',device,'--rounds','50']
            jobs[group]=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while True:
            line=subprocess.check_output(['nvidia-smi','--query-gpu=index,temperature.gpu,utilization.gpu,memory.used,power.draw',
                                          '--format=csv,noheader,nounits'],text=True)
            with (HERE/'gpu_telemetry.csv').open('a') as stream:
                for raw in line.splitlines():
                    fields=[x.strip() for x in raw.split(',')];idx,temp=int(fields[0]),int(fields[1])
                    max_temps[idx]=max(max_temps.get(idx,temp),temp)
                    hot[idx]=hot.get(idx,0)+1 if temp>=82 else 0
                    stream.write(f'{time.time():.3f},'+','.join(fields)+'\n')
                    if temp>=86 or hot[idx]>=3:raise RuntimeError(f'GPU {idx} temperature stop: {temp}C')
            status={g:{'pid':p.pid,'exit_code':p.poll()} for g,p in jobs.items()}
            save(HERE/'job_status.json',{'phase':phase,'supervisor_pid':os.getpid(),'jobs':status,
                                        'max_core_temperatures':max_temps,'updated_unix':time.time()})
            failed=[g for g,p in jobs.items() if p.poll() not in (None,0)]
            if failed:raise RuntimeError('Capture failed: '+','.join(failed))
            if all(p.poll()==0 for p in jobs.values()):break
            time.sleep(5)
        subprocess.run([sys.executable,'-u',str(HERE/'capture_fifty.py'),'--validate','--rounds','50'],env=env,check=True)
        phase='disk_replay'
        save(HERE/'job_status.json',{'phase':phase,'supervisor_pid':os.getpid(),'capture_complete':True,
                                    'paired_state_matches':50,'max_core_temperatures':max_temps,'updated_unix':time.time()})
        pipeline=HERE/'run_fifty_pipeline.py'
        if not pipeline.exists():
            save(HERE/'job_status.json',{'phase':'capture_complete_replay_pending','capture_complete':True,
                 'paired_state_matches':50,'max_core_temperatures':max_temps,'updated_unix':time.time()})
            return
        subprocess.run([sys.executable,'-u',str(pipeline),'--window','5'],env=env,check=True)
        save(HERE/'job_status.json',{'phase':'complete','capture_complete':True,'paired_state_matches':50,
                 'max_core_temperatures':max_temps,'updated_unix':time.time()})
    except Exception as error:
        for process in jobs.values():
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
        save(HERE/'job_status.json',{'phase':'failed','failed_stage':phase,'error':str(error),
                 'max_core_temperatures':max_temps,'updated_unix':time.time()})
        raise
    finally:
        for stream in streams:stream.close()

if __name__=='__main__':main()
