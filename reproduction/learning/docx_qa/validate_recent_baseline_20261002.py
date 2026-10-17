"""Check complete matched trajectories before inserting R4.8 claims."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'r4_tasl_experiments_20260921/recent_baseline_20261002'
rows=[];metadata=[]
for attack in ('none','label_flip'):
 d=B/'results'/attack
 complete=json.loads((d/'complete.json').read_text());assert complete['completed'] and complete['round_records']==150
 env=json.loads((d/'environment.json').read_text());metadata.append(env)
 partition=json.loads((d/'partition.json').read_text())
 flat=[int(i) for inds in partition.values() for i in inds]
 assert len(flat)==60000 and len(set(flat))==60000 and min(flat)==0 and max(flat)==59999
 summary=list(csv.DictReader((d/'summary.csv').open()))
 assert {r['algorithm'] for r in summary}=={'fedavg','tasl_full','lasa'}
 for row in summary:
  result=json.loads((d/f"{row['algorithm']}.json").read_text())
  assert len(result['acc_history'])==len(result['asr_history'])==len(result['round_metrics'])==50
  for values in (result['acc_history'],result['asr_history']):assert all(math.isfinite(v) and 0<=v<=100 for v in values)
  assert [r['round'] for r in result['round_metrics']]==list(range(1,51))
  assert abs(sum(result['acc_history'])/50-float(row['avg_acc']))<1e-10
  assert abs(sum(result['asr_history'])/50-float(row['avg_asr']))<1e-10
  if attack=='none':assert all(v==0 for v in result['asr_history'])
  rows.append({**row,'avg_acc':float(row['avg_acc']),'avg_asr':float(row['avg_asr'])})
assert metadata[0]['initial_model_sha256']==metadata[1]['initial_model_sha256']
assert (B/'results/none/partition.json').read_bytes()==(B/'results/label_flip/partition.json').read_bytes()
assert metadata[0]['config']==metadata[1]['config']
by={(r['algorithm'],r['attack']):r for r in rows}
changes={a:by[a,'none']['avg_acc']-by[a,'label_flip']['avg_acc'] for a in ('fedavg','tasl_full','lasa')}
report={'comment':'R4.8','trajectories':6,'rounds':300,'seed':42,
 'paired_initial_state_and_partition':True,'rows':rows,'clean_minus_LF_Avg_points':changes,
 'LASA_upstream_commit':metadata[0]['LASA_author_commit'],
 'support':'TASL exceeds the measured LASA default configuration in this single matched LF setting.',
 'limits':['One seed and one dataset; no general or statistical superiority over recent methods.',
           'LASA uses author CLI defaults and an aggregation-only adaptation; no outcome-driven tuning.',
           'This comparison has no malicious Executor, DKG, ledger, or secure aggregation deployment.',
           'Results from the 502 environment are kept separate from the five-seed newserver controls.'],
 'exclusion_budget':{'new_clean':0,'new_LF':4,'five_seed_clean':4,
                     'interpretation':'New clean utility tests screening/weighting without permanent exclusion, not its full cost.'},
 'files':[{'path':str(p.relative_to(B)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
          for p in sorted((B/'results').rglob('*')) if p.is_file()]}
(B/'validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
