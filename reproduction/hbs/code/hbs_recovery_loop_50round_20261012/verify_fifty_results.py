"""Independently reread captured, retained and recovered full-state artifacts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import sys
from pathlib import Path
import numpy as np
import torch
from fifty_pipeline_common import HERE,read,write
from storage_bridge import digest,verified_bytes
sys.path.insert(0,str(HERE.parent/'final_protocol_20260928'))
from run_table5_exact_v2 import m
torch.set_num_threads(4)

def equal(a,b):
    if isinstance(a,torch.Tensor):
        return isinstance(b,torch.Tensor) and a.dtype==b.dtype and torch.equal(a,b)
    if isinstance(a,np.ndarray):
        return isinstance(b,np.ndarray) and a.dtype==b.dtype and np.array_equal(a,b)
    if isinstance(a,dict):
        return isinstance(b,dict) and a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):
        return isinstance(b,(list,tuple)) and len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def verify_record(path,row,reference):
    assert digest(path.read_bytes())==row['output_sha256']
    actual=read(path)
    assert actual['committed']==row['expected_commit']
    assert actual['commit_count']==int(row['expected_commit'])
    if actual['committed']:
        state_path=path.with_suffix('.recovered.pt')
        assert digest(state_path.read_bytes())==actual['recovered_state_file_sha256']
        state=torch.load(state_path,weights_only=False,map_location='cpu')
        assert m.protocol_state_hash(state['model'],*state['trust'])==actual['final_state_hash']==reference[row['round']-1]['state_hash']
        assert actual['vs_endorsed'] and actual['vp_endorsed'] and actual['binding_valid']
        assert actual['duplicate_commit_rejected'] and actual['recovery_commitment_matches_capture']
        return 1
    assert not path.with_suffix('.commit.json').exists()
    assert not path.with_suffix('.recovered.pt').exists()
    return 0

def main():
    final=read(HERE/'final_validation.json')
    reference=read(HERE/'capture/noexecutor.json')['round_logs']
    capture=read(HERE/'capture/capture_validation.json')
    for name,value in capture['capture_sha256'].items():
        assert digest((HERE/'capture/snapshots'/name).read_bytes())==value
    assert len(capture['capture_sha256'])==50
    records_checked=0
    recovered_checked=0
    for policy in ('hbs','matched','full'):
        validation=read(HERE/'results'/policy/'validation.json')
        for row in validation['records']:
            recovered_checked+=verify_record(HERE/'results'/policy/f'{row["case"]}_{row["round"]:02d}.json',row,reference)
            records_checked+=1
    chain=read(HERE/'chained_recovery/validation.json')
    for row in chain['records']:
        recovered_checked+=verify_record(HERE/'chained_recovery'/f'round_{row["round"]:02d}.json',row,reference)
        records_checked+=1
    components=[]
    for r in range(1,50):
        state=torch.load(HERE/'chained_recovery'/f'round_{r:02d}.recovered.pt',weights_only=False,map_location='cpu')
        subsequent=torch.load(HERE/'capture/snapshots'/f'round_{r+1:02d}.pt',weights_only=False,map_location='cpu')
        fields=['ema_weights','cos_history','spatial_anchor','temporal_anchor','excluded_clients','violation_counts']
        checks={'round':r,'model_equal':equal(state['model'],subsequent['prior_model'])}
        checks.update({name:equal(state['trust'][i],subsequent['prior_trust_state'][i]) for i,name in enumerate(fields)})
        assert all(value for key,value in checks.items() if key!='round')
        components.append(checks)
    accounting=read(HERE/'retention_accounting.json')
    for policy in accounting['policies']:
        root=HERE/'stores'/policy['policy']
        idx=read(root/'index.json')
        for entry in idx['rounds'].values():
            header=read(root/entry['header'])
            verified_bytes(root/entry['header'],entry['header_sha'])
            verified_bytes(root/entry['governance'],entry['governance_sha'])
            verified_bytes(root/entry['checkpoint'],header['checkpoint_sha'])
            verified_bytes(root/entry['inputs'],header['inputs_sha'])
        current=idx['current_committed']
        verified_bytes(root/current['path'],current['sha256'])
        actual_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
        assert actual_bytes==policy['total_record_and_payload_bytes']+policy['index_bytes']
    for name,value in final['source_hashes'].items():
        assert digest((HERE/name).read_bytes())==value
    reads=read(HERE/'full_checkpoint_reads.json')
    assert reads['verified_reads']==1550 and len(reads['rounds'])==50
    report={'capture_files_sha256_checked':50,'raw_results_sha256_checked':records_checked,
            'recovered_full_state_files_checked':recovered_checked,
            'exact_next_checkpoint_component_checks':components,
            'retention_store_count':3,'all_store_byte_totals_and_hashes_verified':True,
            'verified_checkpoint_reads':1550,'source_hashes_current':True,
            'verifier_sha256':digest(Path(__file__).read_bytes()),'all_passed':True}
    write(HERE/'independent_verification.json',report)
    print({key:value for key,value in report.items() if key!='exact_next_checkpoint_component_checks'},flush=True)

if __name__=='__main__':
    main()
