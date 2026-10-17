"""A fresh process for one round/policy, restoring disk state for each fault."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse
from types import SimpleNamespace
from fifty_pipeline_common import HERE,CASES,read,write,fixture,clean_fixture,check_result
from storage_bridge import digest
from disk_recovery_worker import run
from run_disk_replay import mutate
from fifty_store import recent_rounds

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--policy',choices=['hbs','matched','full'],required=True)
    parser.add_argument('--round',type=int,required=True)
    args=parser.parse_args()
    reference=read(HERE/'capture/noexecutor.json')['round_logs'][args.round-1]
    output=HERE/'results'/args.policy
    output.mkdir(parents=True,exist_ok=True)
    cases=('valid',) if args.policy=='full' else CASES
    rows=[]
    for case in cases:
        temp=output/f'{case}_{args.round:02d}_fixture'
        fixture(HERE/'stores'/args.policy,temp,args.round)
        sha=mutate(temp,case,args.round)
        result=output/f'{case}_{args.round:02d}.json'
        run(SimpleNamespace(store=temp,index_sha=sha,round=args.round,parent=reference['parent_hash'],case=case,output=result))
        expected=case=='valid' or (case=='uniform_prune_no_archive' and args.round in recent_rounds(50,5))
        row=check_result(result,reference,expected)
        row['fixture_index_sha256']=sha
        rows.append(row)
        clean_fixture(temp,output)
    write(output/f'batch_{args.round:02d}.json',rows)

if __name__=='__main__':
    main()
