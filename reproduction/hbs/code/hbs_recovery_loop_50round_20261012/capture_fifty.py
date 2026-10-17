"""Full-data paired 50-round capture; frozen TAS algorithm is not edited."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
from dataclasses import asdict
import hashlib,json,platform,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
FROZEN=HERE.parent/'final_protocol_20260928'
sys.path.insert(0,str(FROZEN))
from run_table5_exact_v2 import common_args,m,SOURCE,write_json
import numpy as np
import torch,torchvision,yaml
EXPECTED='f71067a01c7af357c664306aed944a64c440e53df537a0f2f7e8f0311311e987'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def validate(root,rounds):
    a=json.loads((root/'noexecutor.json').read_bytes())['round_logs']
    b=json.loads((root/'trust_v1v2.json').read_bytes())['round_logs']
    assert len(a)==len(b)==rounds
    assert [x['round'] for x in a]==[x['round'] for x in b]==list(range(1,rounds+1))
    checks={'rounds':rounds,'parent_matches':[x['parent_hash']==y['parent_hash'] for x,y in zip(a,b)],
            'complete_state_matches':[x['state_hash']==y['state_hash'] for x,y in zip(a,b)],
            'all_attacks_rejected':[not bool(x['consensus_passed']) for x in b],
            'both_recovery_endorsements':[bool(x['recovery_v1_endorsed'] and x['recovery_v2_endorsed']) for x in b],
            'capture_sha256':{p.name:sha(p) for p in sorted((root/'snapshots').glob('*.pt'))}}
    write_json(root/'capture_validation.json',checks)
    assert len(checks['capture_sha256'])==rounds
    for key in ('parent_matches','complete_state_matches','all_attacks_rejected','both_recovery_endorsements'):
        assert all(checks[key]),(key,[i+1 for i,x in enumerate(checks[key]) if not x])
    write_json(root/'manifest.json',json.loads((root/'trust_v1v2_manifest.json').read_bytes()))
    print(f'CAPTURE VERIFIED: {rounds}/{rounds} full protocol states',flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--group',choices=['noexecutor','trust_v1v2'])
    p.add_argument('--device',default='cuda:0');p.add_argument('--rounds',type=int,default=50)
    p.add_argument('--validate',action='store_true');p.add_argument('--output',type=Path,default=HERE/'capture')
    args=p.parse_args()
    assert sha(SOURCE)==EXPECTED
    if args.validate:return validate(args.output,args.rounds)
    assert args.group
    args.output.mkdir(parents=True,exist_ok=True)
    out=args.output/(args.group+'.json')
    if out.exists() or (args.output/(args.group+'_manifest.json')).exists():raise FileExistsError(out)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    cfg_path=FROZEN/'configs/table11_fashion_lf.yaml'
    cfg=yaml.safe_load(cfg_path.read_text());cfg['experiment']['rounds']=args.rounds
    cfg['tas']['exclusion']=False
    device=torch.device(args.device)
    common=common_args(cfg,args.rounds,device,smoke_samples=0)
    common['chaos_schedule']={r:m.RoundAttack(True,('A1','A2','A3')[(r-1)%3]) for r in range(1,args.rounds+1)}
    manifest={'purpose':'50-round persisted-state HBS/TAS recovery; full-data functional test',
              'source_sha256':sha(SOURCE),'driver_sha256':sha(__file__),
              'exp2_sha256':sha(FROZEN/'scripts/exp2/exp2_run.py'),'model_sha256':sha(FROZEN/'core/model.py'),
              'configuration':cfg,'group':args.group,'device':args.device,
              'gpu':torch.cuda.get_device_name(device),'python':sys.version,'torch':torch.__version__,
              'torchvision':torchvision.__version__,'numpy':np.__version__,'platform':platform.platform(),
              'deterministic_algorithms':True,'allow_tf32':False,'cpu_threads':4,
              'training_samples':sum(len(x.dataset) for x in common['client_loaders']),
              'test_samples':len(common['test_loader'].dataset),'data_root':os.environ.get('TASL_DATA_ROOT'),
              'attack_schedule':{r:asdict(a) for r,a in common['chaos_schedule'].items()},
              'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    write_json(args.output/(args.group+'_manifest.json'),manifest)
    print('START',args.group,args.device,args.rounds,'rounds',flush=True)
    start=time.perf_counter()
    result=m.run_experiment_group(args.group,**common,round_record_dir=str(args.output/'round_records'),
             snapshot_dir=str(args.output/'snapshots') if args.group=='trust_v1v2' else None)
    write_json(out,asdict(result))
    print('DONE',args.group,'seconds',time.perf_counter()-start,flush=True)

if __name__=='__main__':main()
