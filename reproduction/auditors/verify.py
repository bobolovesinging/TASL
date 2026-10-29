"""Validate released final-auditor records without training or a GPU."""
from pathlib import Path
import collections,hashlib,json,math,statistics
BASE=Path(__file__).resolve().parent
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reject=lambda r:any(r.get(k,False) for k in ['v1_trust','v1_agg','v2_trust','v2_agg'])
def main():
 manifest=read(BASE/'FILE_MANIFEST.json')
 for name,entry in manifest.items():
  p=BASE/name;assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256'],name
 runs=[];overlap=collections.Counter()
 for p in sorted((BASE/'records').rglob('summary.json')):
  s=read(p);rows=read(p.with_name('round_metrics.json'));m=read(p.with_name('manifest.json'));cap=s['cap'];ds=s['dataset'];att=s['attack']
  old=BASE.parent/'governance/records'/('cap0/'+ds+'/matrix/'+ds+'/'+att if cap==0 else 'cap4/matrix/'+ds+'/'+att)
  ref=read(old/'reference/round_metrics.json')
  assert m['reference_sha256']==sha(old/'reference/round_metrics.json') and m['original_manifest_sha256']==sha(old/'vp_only/manifest.json')
  for src in (BASE/'executed_sources').rglob('*.py'):
   name=src.relative_to(BASE/'executed_sources').as_posix()
   if name in ['benchmark_final.py','sampling_final.py']:continue
   assert m['source_sha256'][name]==sha(src),name
  attacked=[r for r in rows if r['attack_active']];honest=[r for r in rows if not r['attack_active']]
  tp=sum(map(reject,attacked));fp=sum(map(reject,honest));matches=sum(a['state_hash']==b['state_hash'] for a,b in zip(rows,ref))
  assert len(rows)==50 and len(attacked)==30 and len(honest)==20
  assert math.isclose(statistics.mean(r['accuracy'] for r in rows),s['avg_accuracy'],abs_tol=1e-10)
  assert tp==s['true_rejections'] and fp==s['false_rejections'] and matches==s['state_matches']
  assert all(not r['skipped'] and r['proposal_binding_valid'] for r in rows)
  assert all(r['recovery_endorsed'] and r['recovery_consistent'] for r in rows if reject(r))
  if cap==0:assert all(not json.loads(r['excluded_clients']) for r in rows)
  if s['group']=='full_tas':
   assert tp==30 and fp==0 and matches==50
   for r in attacked:
    vs=r['v1_trust'] or r['v1_agg'];vp=r['v2_trust'] or r['v2_agg'];overlap['both' if vs and vp else 'vs_only' if vs else 'vp_only' if vp else 'neither']+=1
  runs.append(dict(dataset=ds,attack=att,budget=cap,configuration=s['group'],avg=s['avg_accuracy'],rejected=tp,false_rejections=fp,state_matches=matches))
 assert len(runs)==12
 benchmark=read(BASE/'records/verifier_controls.json');hon=[r for r in benchmark['coverage'] if not r['effective_attack']];tam=[r for r in benchmark['coverage'] if r['effective_attack']]
 assert len(hon)==50 and len(tam)==250
 assert all(r['light_TAS_accepted'] and r['votes']['complete'] for r in hon)
 assert all(not r['light_TAS_accepted'] and not r['votes']['complete'] for r in tam)
 for name,summary in benchmark['cost_summary'].items():
  rows=[r for r in benchmark['observations'] if r['configuration']==name];assert len(rows)==40
  for phase in ['verification','rejected_proposal_audit','recovery']:
   assert math.isclose(statistics.median(r[phase]['total'] for r in rows),summary[phase],abs_tol=1e-10)
  assert math.isclose(statistics.median(r['closed_loop_ms'] for r in rows),summary['closed_loop_ms'],abs_tol=1e-10)
 for name,digest in benchmark['source_sha256'].items():
  relative=name.split('/tas_vp_final_20261029/')[-1] if '/tas_vp_final_20261029/' in name else Path(name).name
  assert sha(BASE/'executed_sources'/relative)==digest,relative
 sampling=read(BASE/'records/sampling_sensitivity.json')
 for name,digest in sampling['source_sha256'].items():
  relative=name.split('/tas_vp_final_20261029/')[-1] if '/tas_vp_final_20261029/' in name else Path(name).name
  assert sha(BASE/'executed_sources'/relative)==digest,relative
 assert sampling['total_subset_proposal_cases']==9207 and sampling['total_honest_false_rejections']==0
 for row in sampling['records']:
  subsets=math.comb(10,row['k']);miss=math.comb(10-row['b'],row['k']) if row['k']<=10-row['b'] else 0
  assert row['subsets']==subsets and row['detected']==subsets-miss and row['honest_rejected']==0
  assert math.isclose(row['rate'],1-miss/subsets,abs_tol=1e-12)
 report=dict(status='passed',verified_files=len(manifest),trajectories=runs,full_tas_same_proposal_overlap=dict(overlap),honest_coverage=len(hon),tampered_coverage=len(tam),sampling_subset_cases=sampling['total_subset_proposal_cases'],costs={k:{n:v for n,v in s.items() if n!='verification_stages'} for k,s in benchmark['cost_summary'].items()})
 print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
