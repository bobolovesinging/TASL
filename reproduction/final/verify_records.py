"""Validate public numerical records without training or tensor payload downloads."""
from pathlib import Path
import hashlib,json,math,statistics
BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text())
def close(a,b):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-9),(a,b)
def main():
 manifest=read(BASE/'FILE_MANIFEST.json')
 for name,entry in manifest.items():
  p=BASE/name;assert p.is_file(),name
  assert p.stat().st_size==entry['bytes'],name
  assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
 records=BASE/'records';report={}
 for condition in ['clean_control','clean_corrected','delayed_control','delayed_corrected']:
  rows=read(records/'c1'/condition/'round_metrics.json');assert len(rows)==50
  if condition.endswith('corrected'):assert all(x['norm_rejected_weight']==0 for x in rows)
  close(statistics.mean(x['accuracy'] for x in rows),read(records/'c1'/condition/'summary.json')['average_accuracy'])
 clean0=read(records/'c1/clean_control/round_metrics.json');clean1=read(records/'c1/clean_corrected/round_metrics.json')
 assert all(a['state_hash']==b['state_hash'] for a,b in zip(clean0,clean1))
 summaries={}
 for attack in ['none','label_flip']:
  for method in ['fedavg','fltrust','tasl_full','lasa']:
   rows=[]
   for seed in [42,123,2024,3407,7711]:
    p=records/'c6_repeated'/f'{method}_{attack}_seed{seed}';metrics=read(p/'metrics.json');summary=read(p/'summary.json')
    assert len(metrics['acc_history'])==len(metrics['round_metrics'])==50
    close(statistics.mean(metrics['acc_history']),summary['avg']);close(metrics['acc_history'][-1],summary['final'])
    if attack=='label_flip':close(statistics.mean(metrics['asr_history']),summary['lf_asr'])
    rows.append(summary)
    for source in ['scripts/exp2/exp2_run.py','core/model.py']:
     assert read(p/'manifest.json')['source_sha256'][source]==hashlib.sha256((BASE/source).read_bytes()).hexdigest()
   summaries[f'{method}_{attack}']={k:dict(mean=statistics.mean(x[k] for x in rows),sd=statistics.stdev(x[k] for x in rows)) for k in ['avg','final']+(['lf_asr'] if attack=='label_flip' else [])}
 for key,values in summaries.items():
  for metric,values2 in values.items():
   for stat,v in values2.items():close(v,read(records/'c6_statistics.json')[key][metric][stat])
 reference=read(records/'c2/reference/round_metrics.json');audited=read(records/'c2/full_tas/round_metrics.json');learning=read(records/'c2/learning_entry_rounds.json')
 assert len(reference)==len(audited)==len(learning)==50
 assert all(x['state_hash']==y['state_hash']==z['state_hash'] for x,y,z in zip(reference,audited,learning))
 assert read(records/'c2/full_tas/summary.json')['true_rejections']==30
 assert read(records/'c2/full_tas/summary.json')['false_rejections']==0
 for source,digest in read(records/'c2/reference/manifest.json')['source_sha256'].items():
  assert hashlib.sha256((BASE/source).read_bytes()).hexdigest()==digest,source
 for seed in [42,123,2024,3407,7711]:
  partitions=read(records/'partitions'/f'seed{seed}.json')['training_indices'];indices=[i for v in partitions.values() for i in v]
  assert len(indices)==len(set(indices))==60000 and min(indices)==0 and max(indices)==59999
 hbs=read(records/'hbs_final_v2/summary.json');inventory=read(records/'hbs_final_v2/inventory.json')
 assert sum(x['compact_bytes'] for x in inventory)==hbs['compact_total_bytes']
 total=hbs['policies']['full']['total_bytes']
 for d in hbs['policies'].values():
  assert d['active_bytes']+d['archive_bytes']==total
  close(d['active_reduction_pct'],100*(1-d['active_bytes']/total))
 tests=read(records/'hbs_final_v2/recovery_tests.json');normal=[x for x in tests if x['case']=='normal']
 assert len(normal)==50 and all(x['accepted'] and x['matches_reference'] for x in normal)
 faults=[x for x in tests if x['case'] not in ['normal','duplicate_commit','archive_unavailable']]
 assert len(faults)==450 and all(not x['accepted'] for x in faults)
 numerical=read(records/'c7/summary.json');assert numerical['round_records']==150 and numerical['honest_rejections']==0
 raw=[json.loads(line) for line in (records/'c7/rounds.jsonl').read_text().splitlines()]
 assert len(raw)==150
 for key in ['max_decode_abs_error','max_weight_abs_error','max_cosine_abs_error','max_direction_abs_error']:
  close(max(x[key] for x in raw),numerical[key])
 feedback=read(records/'feedback/ckks_summary.json');assert len(feedback['runs'])==3
 assert all(x['attacked_rejected']==30 and x['honest_acceptance']==20 for x in feedback['runs'])
 for x in feedback['runs']:
  rounds=read(records/'feedback'/f"ckks_{x['repeat']}.json")
  close(statistics.mean(r['accuracy'] for r in rounds),x['avg']);close(rounds[-1]['accuracy'],x['final'])
 report=dict(files_verified=len(manifest),five_seed_training_runs=40,paired_complete_states=50,learning_complete_states=50,
  indexed_training_assignments=300000,hbs_recovery_matches=50,hbs_faults_blocked=450,decoded_replay_rounds=150,
  decoded_feedback_rounds=150,statistics=summaries,
  scope='Recorded numerical-output verification; not a fresh training, network, threshold-decryption, or tensor-file integrity check.')
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
