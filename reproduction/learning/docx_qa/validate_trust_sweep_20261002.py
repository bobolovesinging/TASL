import csv,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
B=ROOT/'r4_tasl_experiments_20260921/recent_baseline_20261002'
rows=[]
for group,count in [('exposed',6),('anchor_norm',4)]:
 d=B/'trust_sweep'/group;meta=json.loads((d/'complete.json').read_text())
 assert meta['cases']==count and meta['default_transform_equivalence'] and meta['seed']==42
 saved=list(csv.DictReader((d/'summary.csv').open()));assert len(saved)==count
 for row in saved:
  q=json.loads((d/f"{row['parameter']}_{float(row['value'])}.json").read_text())
  r=q['result']; assert len(r['round_metrics'])==len(r['acc_history'])==50
  assert all(math.isfinite(v) and 0<=v<=100 for v in r['acc_history']+r['asr_history'])
  assert abs(sum(r['acc_history'])/50-float(row['avg_acc']))<1e-10
  assert abs(sum(r['asr_history'])/50-float(row['avg_asr']))<1e-10
  zeros=sum(t['honest_rejected'] for t in r['round_metrics'])
  assert zeros==int(row['honest_zero_weights']) and int(row['honest_client_rounds'])==300
  rows.append({'parameter':row['parameter'],'value':float(row['value']),
               'avg_acc':float(row['avg_acc']),'avg_asr':float(row['avg_asr']),
               'honest_zero_count':zeros,'honest_zero_percent':100*zeros/300})
base=json.loads((B/'results/label_flip/tasl_full.json').read_text())
report={'comment':'R4.6','trajectories':10,'round_records':500,'seed':42,'rows':rows,
 'default':{'avg_acc':base['avg_acc'],'avg_asr':base['avg_asr'],
            'honest_zero_count':sum(r['honest_rejected'] for r in base['round_metrics']),
            'honest_zero_percent':sum(r['honest_rejected'] for r in base['round_metrics'])/3},
 'support':'One-factor heuristic sensitivity and honest-client screening cost in the measured LF setting.',
 'limits':['Not an optimizer or convergence proof; defaults are not selected from the test results.',
           'One training seed, no clean scan and no interaction grid.',
           'Changing the anchor fraction changes all three anchor blend occurrences together.',
           'Zero client weights are not honest Executor false rejections; fixed Vp sample count is not scanned.']}
(B/'trust_sweep/validation.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
