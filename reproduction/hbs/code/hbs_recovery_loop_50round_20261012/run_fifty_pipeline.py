"""Fifty actual rounds: persistent recovery, archives, controls and verified reads."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import concurrent.futures
import csv
import io
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import torch
from fifty_pipeline_common import HERE,read,write,fixture,clean_fixture,make_comparator,check_result
from run_disk_replay import prepare,serialize
from storage_bridge import digest,verified_bytes,validate_checkpoint

torch.set_num_threads(4)
torch.set_num_interop_threads(1)

def status(phase,**extra):
    value=read(HERE/'job_status.json')
    value.update(phase=phase,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra)
    temporary=HERE/'job_status.tmp.json'
    write(temporary,value)
    temporary.replace(HERE/'job_status.json')
    print(json.dumps({'phase':phase,**extra}),flush=True)

def command_run(command,log):
    with log.open('w',encoding='utf-8') as stream:
        completed=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
    if completed.returncode:
        raise RuntimeError(f'Child failed: {log}')

def batch(policy,r):
    command_run([sys.executable,str(HERE/'run_fifty_batch.py'),'--policy',policy,'--round',str(r)],
                HERE/'logs'/f'{policy}_{r:02d}.log')
    return policy,r,read(HERE/'results'/policy/f'batch_{r:02d}.json')

def chain(reference):
    root=HERE/'chained_recovery'
    root.mkdir(exist_ok=False)
    previous=None
    rows=[]
    for row in reference:
        r=row['round']
        temp=root/f'round_{r:02d}_fixture'
        fixture(HERE/'stores/hbs',temp,r)
        index=read(temp/'index.json')
        entry=index['rounds'][str(r)]
        if previous is not None:
            prior=torch.load(previous,weights_only=False,map_location='cpu')
            checkpoint={'schema':'tasl-full-state-v1','round':r,'model':prior['model'],'trust':prior['trust']}
            cp=temp/entry['checkpoint']
            cp.write_bytes(serialize(checkpoint))
            header=read(temp/entry['header'])
            header['checkpoint_sha']=digest(cp.read_bytes())
            write(temp/entry['header'],header)
            entry['header_sha']=digest((temp/entry['header']).read_bytes())
            write(temp/'index.json',index)
        output=root/f'round_{r:02d}.json'
        command_run([sys.executable,str(HERE/'disk_recovery_worker.py'),'--store',str(temp),
                     '--index-sha',digest((temp/'index.json').read_bytes()),'--round',str(r),
                     '--parent',row['parent_hash'],'--case','chained_valid','--output',str(output)],
                     HERE/'logs'/f'chain_{r:02d}.log')
        check=check_result(output,row,True)
        check['prior_loaded_from_previous_recovered_file']=r>1
        rows.append(check)
        previous=output.with_suffix('.recovered.pt')
        clean_fixture(temp,root)
        write(root/'partial_validation.json',rows)
        if r%5==0:
            status('chained_recovery',chain_rounds_completed=r)
    write(root/'validation.json',{'rounds':50,'complete_hash_matches':sum(x['passed'] for x in rows),'records':rows})
    return previous,rows

def persist_current(current,reference):
    # The current accepted W50 is additional to five retained parent/input pairs.
    value=torch.load(current,weights_only=False,map_location='cpu')
    blob=serialize({'schema':'tasl-accepted-state-v1','accepted_round':50,
                    'model':value['model'],'trust':value['trust']})
    for policy in ('hbs','matched','full'):
        root=HERE/'stores'/policy
        destination='active/current_accepted_state.pt' if policy=='hbs' else 'objects/current_accepted_state.pt'
        (root/destination).write_bytes(blob)
        idx=read(root/'index.json')
        idx['current_committed']={'path':destination,'sha256':digest(blob),'accepted_round':50,
                                  'protocol_state_hash':reference[-1]['state_hash']}
        write(root/'index.json',idx)
    return len(blob)

def inventory():
    rows=[]
    for policy in ('hbs','matched','full'):
        root=HERE/'stores'/policy
        idx=read(root/'index.json')
        active,archive,compact=0,0,0
        entries=[]
        for r,e in idx['rounds'].items():
            header=read(root/e['header'])
            for key,sha_key in (('checkpoint','checkpoint_sha'),('inputs','inputs_sha')):
                path=root/e[key]
                assert digest(path.read_bytes())==header[sha_key]
                archived=e[key].startswith('archive/')
                size=path.stat().st_size
                archive+=size if archived else 0
                active+=0 if archived else size
                entries.append({'round':int(r),'kind':key,'bytes':size,'archive':archived,'sha256':header[sha_key]})
            compact+=sum((root/e[key]).stat().st_size for key in ('header','governance'))
        current=idx['current_committed']
        assert digest((root/current['path']).read_bytes())==current['sha256']
        current_bytes=(root/current['path']).stat().st_size
        active+=current_bytes
        row={'policy':policy,'permanent_compact_bytes':compact,'active_heavy_bytes':active,
             'archive_heavy_bytes':archive,'active_record_and_payload_bytes':active+compact,
             'total_record_and_payload_bytes':active+archive+compact,'index_bytes':(root/'index.json').stat().st_size,
             'current_accepted_checkpoint_bytes':current_bytes,'retained_parent_input_pairs':50 if policy=='full' else 5,
             'archived_parent_input_pairs':0 if policy=='full' else 45,'entries':entries}
        rows.append(row)
    assert rows[0]['active_record_and_payload_bytes']==rows[1]['active_record_and_payload_bytes']
    assert len({row['total_record_and_payload_bytes'] for row in rows})==1
    write(HERE/'retention_accounting.json',{'scope':'50 parent model+six-field trust checkpoints, 50 same-round update/proposal payloads, 100 compact records, one current accepted W50; single archive copy; policy indexes reported separately; test fixtures and training capture excluded','policies':rows})
    return rows

def read_probes():
    sys.path.insert(0,str(HERE.parent/'final_protocol_20260928'))
    from run_table5_exact_v2 import m
    store=HERE/'stores/hbs'
    idx=read(store/'index.json')
    rows=[]
    samples=[]
    for r,e in idx['rounds'].items():
        header=read(store/e['header'])
        path=store/e['checkpoint']
        # One warm-up, then 31 verified read+deserialize samples; not a cold-I/O benchmark.
        torch.load(io.BytesIO(verified_bytes(path,header['checkpoint_sha'])),map_location='cpu',weights_only=False)
        latencies=[]
        for trial in range(31):
            begin=time.perf_counter_ns()
            checkpoint=torch.load(io.BytesIO(verified_bytes(path,header['checkpoint_sha'])),map_location='cpu',weights_only=False)
            elapsed=(time.perf_counter_ns()-begin)/1e6
            validate_checkpoint(checkpoint,int(r))
            assert m.protocol_state_hash(checkpoint['model'],*checkpoint['trust'])==header['parent_hash']
            latencies.append(elapsed)
            samples.append({'round':int(r),'sample':trial+1,'read_verify_deserialize_ms':elapsed})
        rows.append({'round':int(r),'location':'archive' if e['checkpoint'].startswith('archive/') else 'active',
                     'checkpoint_bytes':path.stat().st_size,'history_length':len(checkpoint['trust'][1]),
                     'median_ms':statistics.median(latencies),'samples_ms':latencies,'all_complete_parent_hashes_match':True})
        if int(r)%10==0:
            status('verified_checkpoint_reads',read_rounds_completed=int(r))
    write(HERE/'full_checkpoint_reads.json',{'actual_rounds':50,'samples_per_round':31,'verified_reads':len(samples),'overall_median_ms':statistics.median(x['read_verify_deserialize_ms'] for x in samples),'scope':'warm OS-cache read+SHA256 verification+torch deserialization; complete-state hash checked outside timing; local disk, no network','rounds':rows})
    with (HERE/'checkpoint_read_samples.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(samples[0]));writer.writeheader();writer.writerows(samples)
    return len(samples)

def main():
    started=time.monotonic()
    capture=read(HERE/'capture/capture_validation.json')
    assert capture['rounds']==50 and all(capture['complete_state_matches'])
    assert all(capture['parent_matches']) and all(capture['all_attacks_rejected'])
    assert all(capture['both_recovery_endorsements'])
    (HERE/'stores').mkdir(exist_ok=False)
    (HERE/'logs').mkdir(exist_ok=False)
    status('preparing_real_retention_stores')
    reference=prepare(HERE/'capture',HERE/'stores/hbs')
    assert len(reference)==50
    make_comparator(HERE/'stores/hbs',HERE/'stores/matched',True)
    make_comparator(HERE/'stores/hbs',HERE/'stores/full',False)
    records={'hbs':[],'matched':[],'full':[]}
    jobs=[(policy,r) for r in range(1,51) for policy in ('hbs','matched','full')]
    status('disk_fault_replay',total_policy_round_batches=len(jobs),completed_policy_round_batches=0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        pending=[pool.submit(batch,*job) for job in jobs]
        completed=0
        for future in concurrent.futures.as_completed(pending):
            policy,r,checks=future.result()
            records[policy].extend(checks)
            completed+=1
            if completed%10==0 or completed==len(jobs):
                status('disk_fault_replay',completed_policy_round_batches=completed,passed_cases=sum(len(x) for x in records.values()))
    for policy,checks in records.items():
        checks.sort(key=lambda x:(x['round'],x['case']))
        write(HERE/'results'/policy/'validation.json',{'cases':len(checks),'passed':sum(x['passed'] for x in checks),
             'commits':sum(x['committed'] for x in checks),'negative_cases_blocked':sum(not x['committed'] for x in checks),'records':checks})
        with (HERE/'results'/policy/'case_summary.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(checks[0]));writer.writeheader();writer.writerows(checks)
    status('chained_recovery',chain_rounds_completed=0)
    current,chain_rows=chain(reference)
    persist_current(current,reference)
    policies=inventory()
    verified_reads=read_probes()
    summary={'actual_training_rounds':50,'seed':42,'paired_training_state_matches':50,
             'window':5,'hbs_cases':len(records['hbs']),'hbs_cases_passed':sum(x['passed'] for x in records['hbs']),
             'matched_cases':len(records['matched']),'matched_cases_passed':sum(x['passed'] for x in records['matched']),
             'hbs_commits':sum(x['committed'] for x in records['hbs']),
             'hbs_negative_cases_blocked':sum(not x['committed'] for x in records['hbs']),
             'full_retention_matches':sum(x['passed'] for x in records['full']),
             'chained_complete_state_matches':sum(x['passed'] for x in chain_rows),
             'verified_checkpoint_reads':verified_reads,'retention_policies':[{k:v for k,v in x.items() if k!='entries'} for x in policies],
             'pipeline_seconds':time.monotonic()-started,
             'source_hashes':{p.name:digest(p.read_bytes()) for p in HERE.glob('*.py')},
             'purpose':['R4.1: semantic retention serves audit/recovery','R4.5: full-state dual-endorsed recovery from disk','R4.14: measured 50-round payloads, archive retrieval, matched retention backend'],
             'limits':['one seed, one Fashion-MNIST trajectory; fault cases are not independent training replicates','single-machine plaintext trusted-manifest experiment, not deployed blockchain/IPFS/DKG/signature/network/gas benchmark','matched comparator uses same record bytes and frozen TAS worker to isolate retention; no superior compression claim','current checkpoint and growing trust history included; policy indexes listed separately']}
    write(HERE/'final_validation.json',summary)
    status('complete',replay_complete=True,final_validation_sha256=digest((HERE/'final_validation.json').read_bytes()),pipeline_seconds=summary['pipeline_seconds'])

if __name__=='__main__':
    try:
        main()
    except Exception:
        (HERE/'pipeline_error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        status('pipeline_failed',error_file='pipeline_error.txt')
        raise
