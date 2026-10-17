"""Retention-only extension; the frozen TAS verifier is not modified."""
import copy
import json
from pathlib import Path
import shutil
from storage_bridge import digest, json_bytes
from checkpoint_prune import retain_flat_objects
from fifty_store import recent_rounds

HERE=Path(__file__).resolve().parent
CASES=('valid','missing_updates','corrupt_updates','corrupt_checkpoint',
       'incomplete_trust','wrong_round','wrong_parent','missing_vs_vote',
       'missing_vp_vote','corrupt_replacement','uniform_prune_no_archive')

def read(path):
    return json.loads(path.read_bytes())

def write(path,value):
    path.write_bytes(json_bytes(value))

def fixture(source,destination,r):
    # Only copy the selected round, never hard-link a mutable fault fixture.
    destination.mkdir(exist_ok=False)
    original=read(source/'index.json')
    entry=copy.deepcopy(original['rounds'][str(r)])
    for key in ('header','governance','checkpoint','inputs'):
        target=destination/entry[key]
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source/entry[key],target)
    (destination/'archive').mkdir(exist_ok=True)
    original['rounds']={str(r):entry}
    original.pop('current_committed',None)
    write(destination/'index.json',original)
    return digest((destination/'index.json').read_bytes())

def clean_fixture(path,parent):
    target=path.resolve()
    if target.parent!=parent.resolve() or not target.name.endswith('_fixture'):
        raise ValueError('Unsafe disposable-fixture cleanup')
    shutil.rmtree(target)

def make_comparator(hbs,root,prune):
    root.mkdir(exist_ok=False)
    for name in ('objects','archive','records'):
        (root/name).mkdir()
    idx=read(hbs/'index.json')
    keep={}
    for r,entry in idx['rounds'].items():
        for key in ('header','governance'):
            blob=(hbs/entry[key]).read_bytes()
            dest='records/'+digest(blob)+'.json'
            (root/dest).write_bytes(blob)
            entry[key]=dest
        for key in ('checkpoint','inputs'):
            blob=(hbs/entry[key]).read_bytes()
            name=digest(blob)+'.pt'
            (root/'objects'/name).write_bytes(blob)
            entry[key]='objects/'+name
            keep[name]=(int(r) in recent_rounds(len(idx['rounds']),idx['window'])) if prune else True
    mapping=retain_flat_objects(root,keep)
    for entry in idx['rounds'].values():
        for key in ('checkpoint','inputs'):
            entry[key]=mapping[Path(entry[key]).name]
    idx['backend']='flat checkpoint-and-prune' if prune else 'full retention'
    write(root/'index.json',idx)

def check_result(output,reference,expected):
    result=read(output)
    matches=result.get('final_state_hash')==reference['state_hash']
    no_side_effects=not output.with_suffix('.commit.json').exists() and not output.with_suffix('.recovered.pt').exists()
    valid=(result['committed']==expected and result['commit_count']==int(expected))
    if expected:
        valid=valid and matches and result.get('duplicate_commit_rejected') and result.get('vs_endorsed') and result.get('vp_endorsed') and result['recovery_consistency']['consistent'] and result['recovery_commitment_matches_capture']
    else:
        valid=valid and no_side_effects and result.get('final_state_hash') in (None,reference['parent_hash'])
    row={'case':result['case'],'round':reference['round'],'expected_commit':expected,
         'committed':result['committed'],'commit_count':result['commit_count'],
         'final_matches_noexecutor':matches,'negative_has_no_commit_or_state_file':no_side_effects if not expected else None,
         'original_rejected':result.get('original_proposal_rejected'),
         'vs_endorsed':result.get('vs_endorsed'),'vp_endorsed':result.get('vp_endorsed'),
         'local_loop_ms':result['local_loop_ms'],'error':result.get('error'),
         'worker_pid':result['pid'],'output_sha256':digest(output.read_bytes()),'passed':bool(valid)}
    if not valid:
        raise AssertionError(row)
    return row
