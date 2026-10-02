"""Validate independent runs; use seeds, not proposal opportunities, for uncertainty."""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path

SEEDS=(42,123,2024,3407,7711)
GROUPS=('trust_v1','trust_v2')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def b(row,key): return bool(int(row[key]))
def stats(values):
 mean=statistics.mean(values);sd=statistics.stdev(values)
 half=2.7764451051977987*sd/math.sqrt(len(values))
 return {'mean':mean,'sample_sd':sd,'mean_95pct_t_interval':[mean-half,mean+half]}
def validate_case(root,prior,attack,seed):
 directory=root/'results'/f'{attack}_seed{seed}'
 raw=directory/'tasv2_mixed12_r50_b4.csv';summary=directory/'tasv2_summary.csv'
 data=rows(raw); reported={r['group']:r for r in rows(summary)}
 assert len(data)==100 and set(reported)==set(GROUPS),(attack,seed,'incomplete')
 baseline=rows(prior/'results'/f'{attack}_seed{seed}'/'tasv2_mixed12_r50_b4.csv')
 reference={int(r['round']):r for r in baseline if r['group']=='noexecutor'}
 full={int(r['round']):r for r in baseline if r['group']=='trust_v1v2'}
 assert len(reference)==len(full)==50
 output=[]
 for group in GROUPS:
  series={int(r['round']):r for r in data if r['group']==group}
  assert sorted(series)==list(range(1,51))
  journal=directory/'round_records/mixed12'/f'{group}.jsonl'
  logs=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else None
  if logs is not None:assert len(logs)==50
  for number,row in series.items():
   if logs is not None:
    record=logs[number-1]
    assert record['round']==number and record['group']==group
    for field in ('state_hash','parent_hash','consensus_passed','accuracy','attack_active','healed_applied','honest_rejection'):
     assert str(row[field])==str(record[field]),(attack,seed,group,number,field)
   assert math.isfinite(float(row['accuracy']))
   assert json.loads(row['excluded_clients'])==[]
   assert b(row,'proposal_binding_valid')
   assert row['attack_active']==full[number]['attack_active'] and row['attack_type']==full[number]['attack_type']
   if number>1:assert row['parent_hash']==series[number-1]['state_hash']
   detection=b(row,'v1_trust') or b(row,'v1_agg') or b(row,'v2_trust') or b(row,'v2_agg')
   assert b(row,'consensus_passed') != detection
   assert b(row,'honest_rejection')==(not b(row,'attack_active') and detection)
   if group=='trust_v1':assert not b(row,'v2_trust') and not b(row,'v2_agg')
   else:assert not b(row,'v1_trust') and not b(row,'v1_agg')
   if detection:
    assert b(row,'recovery_attempted')
    # These are diagnostic correctness checks, not requirements on recall.
    if b(row,'recovery_endorsed') and b(row,'recovery_consistent'):
     assert b(row,'healed_applied') and not b(row,'skipped')
     assert float(row['committed_relative_error'])<=1.e-6
   else:assert not b(row,'recovery_attempted') and not b(row,'healed_applied')
  attacked=[r for r in series.values() if b(r,'attack_active')]
  honest=[r for r in series.values() if not b(r,'attack_active')]
  assert len(attacked)==30 and len(honest)==20
  tp=sum(not b(r,'consensus_passed') for r in attacked)
  fp=sum(not b(r,'consensus_passed') for r in honest)
  a1=[r for r in attacked if r['attack_type']=='A1'];a2=[r for r in attacked if r['attack_type']=='A2']
  assert len(a1)==10 and len(a2)==20
  avg=statistics.mean(float(r['accuracy']) for r in series.values())
  assert abs(avg-float(reported[group]['avg_acc']))<0.001
  assert int(reported[group]['blocked_rounds'])==tp
  assert int(reported[group]['attack_rounds'])==30
  output.append({'attack':attack,'seed':seed,'group':group,'avg_accuracy_percent':avg,
   'full_avg_accuracy_percent':statistics.mean(float(r['accuracy']) for r in full.values()),
   'tp':tp,'fp':fp,'fn':30-tp,'tn':20-fp,
   'recall_percent':100*tp/30,'precision_percent':100*tp/(tp+fp) if tp+fp else None,
   'honest_acceptance_percent':100*(20-fp)/20,'false_rejection_percent':100*fp/20,
   'a1_detected':sum(not b(r,'consensus_passed') for r in a1),
   'a2_detected':sum(not b(r,'consensus_passed') for r in a2),
   'recovered':sum(b(r,'healed_applied') for r in series.values()),
   'skipped':sum(b(r,'skipped') for r in series.values()),
   'state_accuracy_matches_reference':sum(r['state_hash']==reference[n]['state_hash'] and r['accuracy']==reference[n]['accuracy'] for n,r in series.items()),
   'initial_parent_matches_reference':series[1]['parent_hash']==reference[1]['parent_hash'],
   'first_round_update_hashes_match_reference':series[1]['update_hashes']==reference[1]['update_hashes'],
   'csv_sha256':digest(raw),'summary_sha256':digest(summary),
   'jsonl_sha256':digest(journal) if journal.exists() else None,
   'record_format':'Complete per-round CSV from the frozen source main() CLI; no separate live JSONL.'})
 return output
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--attack',choices=['lf','scaling'])
 args=p.parse_args();root=args.root
 prior=root.parent/'fashion_fixed_five_seed_20260930'
 attacks=[args.attack] if args.attack else ['lf','scaling']
 plan=json.loads((root/'plan_manifest.json').read_text())
 for cfg in plan['configurations']:
  assert digest(root/'configs'/cfg['file'])==cfg['sha256']
 cases=[r for a in attacks for s in SEEDS for r in validate_case(root,prior,a,s)]
 pooled=[]
 for attack in attacks:
  status=json.loads((root/f'{attack}_status.json').read_text());assert status['state']=='completed'
  assert [r['seed'] for r in status['completed']]==list(SEEDS)
  assert status['source_sha256']==plan['source_sha256']
  for group in GROUPS:
   subset=[r for r in cases if r['attack']==attack and r['group']==group]
   tp=sum(r['tp'] for r in subset);fp=sum(r['fp'] for r in subset)
   difference=[r['full_avg_accuracy_percent']-r['avg_accuracy_percent'] for r in subset]
   ds=stats(difference);ds['paired_dz']=ds['mean']/ds['sample_sd'] if ds['sample_sd'] else None
   pooled.append({'attack':attack,'group':group,'attacked':150,'honest':100,'tp':tp,'fp':fp,'fn':150-tp,'tn':100-fp,
    'recall_percent':100*tp/150,'precision_percent':100*tp/(tp+fp) if tp+fp else None,
    'honest_acceptance_percent':100-fp,'false_rejection_percent':fp,
    'a1_detected':sum(r['a1_detected'] for r in subset),'a1_total':50,
    'a2_detected':sum(r['a2_detected'] for r in subset),'a2_total':100,
    'recall_seed_stats':stats([r['recall_percent'] for r in subset]),
    'avg_accuracy_seed_stats':stats([r['avg_accuracy_percent'] for r in subset]),
    'full_minus_single_auditor_paired_avg_accuracy':ds,
    'recovered':sum(r['recovered'] for r in subset),'skipped':sum(r['skipped'] for r in subset),
    'state_accuracy_matches_reference':sum(r['state_accuracy_matches_reference'] for r in subset)})
 result={'validation':'passed','comments':['R4.9','R2.2','R2.4'],'cases':cases,'pooled':pooled,
 'statistical_unit':'Five independent training seeds; proposals are pooled descriptive opportunities, not independent runs.',
 'limitations':'Single host and fixed tested A1/A2 schedule; independent Vs/Vp trajectories cannot be used to infer their joint detection overlap.'}
 output=root/('validation.json' if not args.attack else f'validation_{args.attack}.json')
 output.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({'validation':'passed','cases':len(cases),'pooled':pooled},indent=2))
if __name__=='__main__':main()
