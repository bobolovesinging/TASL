"""Verify final Table 4 V_P rows using the standard library only."""
from pathlib import Path
import ast,hashlib,json,math,statistics
BASE=Path(__file__).resolve().parent
def main():
 read=lambda p:json.loads(p.read_text(encoding='utf-8'));sha=lambda x:hashlib.sha256(x).hexdigest()
 manifest=read(BASE/'FILE_MANIFEST.json')
 for n,x in manifest.items():
  raw=(BASE/n).read_bytes();assert len(raw)==x['bytes'] and sha(raw)==x['sha256'],n
 sources=read(BASE/'records/normalized_sources.json');runs=read(BASE/'records/normalized_vp_runs.json');assert len(runs)==4
 for scope,files in sources.items():
  for name,text in files.items():
   if name.endswith('.py'):compile(text,name,'exec')
 results=[]
 for run in runs:
  rows=run['round_metrics'];m=run['manifest'];s=run['summary'];files=sources[run['source_scope']]
  for n,h in m['source_sha256'].items():assert sha(files[n].encode('utf-8'))==h,n
  assert m['configuration']==json.loads(files['matrix_config.json'])
  ref=read(BASE.parent/'governance/records/cap0'/run['dataset']/'matrix'/run['dataset']/run['attack']/'reference/round_metrics.json')
  assert len(rows)==50 and [r['round'] for r in rows]==list(range(1,51))
  reject=lambda r:bool(r['v2_trust'] or r['v2_agg'])
  attacked=[r for r in rows if r['attack_active']];honest=[r for r in rows if not r['attack_active']]
  assert len(attacked)==30 and len(honest)==20
  tp=sum(map(reject,attacked));fp=sum(map(reject,honest));avg=statistics.mean(r['accuracy'] for r in rows)
  assert tp==s['true_rejections'] and fp==s['false_rejections'] and math.isclose(avg,s['avg_accuracy'],abs_tol=1e-10)
  for r in rows:
   assert not r['skipped'] and r['proposal_binding_valid'] and not json.loads(r['excluded_clients'])
   if reject(r):assert r['healed_applied'] and r['recovery_endorsed'] and r['recovery_consistent']
   if r['attack_active']:
    truth,pub,used=(json.loads(r[k]) for k in ['final_weights','published_weights','used_weights'])
    err=max(abs(pub[c]-truth[c]) for c in truth);assert err>1e-6 or r['proposal_relative_error']>1e-6
    if r['attack_type']=='A1':assert pub==used and abs(sum(pub.values())-1)<1e-12 and err>1e-6
  results.append(dict(dataset=run['dataset'],attack=run['attack'],avg=avg,rejected=tp,false_rejections=fp,state_matches=sum(x['state_hash']==y['state_hash'] for x,y in zip(rows,ref))))
 assert sum(r['rejected'] for r in results)==111 and sum(r['false_rejections'] for r in results)==0
 assert sum(r['state_matches'] for r in results)==109
 report=dict(status='passed',rows=results,verified_files=len(manifest));print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
