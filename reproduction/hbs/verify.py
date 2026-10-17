from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_bytes())
manifest=read(root/'MANIFEST.json')
for name,expected in manifest['files'].items():
    p=root/name
    assert p.stat().st_size==expected['bytes'],name
    assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name
r=root/'records'
a=read(r/'capture/noexecutor.json')['round_logs']
b=read(r/'capture/trust_v1v2.json')['round_logs']
assert len(a)==len(b)==50
assert all(x['parent_hash']==y['parent_hash'] and x['state_hash']==y['state_hash'] for x,y in zip(a,b))
assert all(not y['consensus_passed'] and y['recovery_v1_endorsed'] and y['recovery_v2_endorsed'] for y in b)
count=0
for policy,size,commits,blocked in [('hbs',550,55,495),('matched',550,55,495),('full',50,50,0)]:
    data=read(r/'results'/policy/'validation.json')
    assert data['cases']==data['passed']==size
    assert data['commits']==commits and data['negative_cases_blocked']==blocked
    for row in data['records']:
        p=r/'results'/policy/f"{row['case']}_{row['round']:02d}.json"
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['output_sha256']
        assert row['passed'];count+=1
chain=read(r/'chained_recovery/validation.json')
assert chain['rounds']==chain['complete_hash_matches']==50
for row in chain['records']:
    p=r/'chained_recovery'/f"round_{row['round']:02d}.json"
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['output_sha256']
    assert row['passed'];count+=1
assert count==1200
policies=read(r/'retention_accounting.json')['policies']
assert all(p['total_record_and_payload_bytes']==2039961654 for p in policies)
assert policies[0]['active_record_and_payload_bytes']==policies[1]['active_record_and_payload_bytes']==219555200
reads=read(r/'full_checkpoint_reads.json')
assert reads['verified_reads']==1550 and len(reads['rounds'])==50
assert all(p['all_complete_parent_hashes_match'] for p in reads['rounds'])
verification=read(r/'independent_verification.json')
assert verification['all_passed'] and verification['recovered_full_state_files_checked']==210
assert len(verification['exact_next_checkpoint_component_checks'])==49
print('PASS:',len(manifest['files']),'file hashes; 1200 raw-result links; 50-round paired and chained states; retention totals; 1550 recorded verified reads')
