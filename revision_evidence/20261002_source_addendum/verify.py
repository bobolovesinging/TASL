"""Offline source-row and file-integrity verification, not retraining."""
import csv,hashlib,json
from pathlib import Path

def main():
 root=Path(__file__).resolve().parent
 manifest=json.loads((root/'manifest.json').read_text())
 for x in manifest['files']:
  p=(root/x['path']).resolve();assert root in p.parents
  assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256'],x['path']
 data=json.loads((root/'matched_records.json').read_text())
 approved=json.loads((root/'workbook_seed_records.json').read_text())
 targets={(x['algorithm'],x['ratio_percent'],x['seed']):x for x in approved['seed_level_records']}
 aliases={'fedavg':'FedAvg','multi_krum':'Multi-Krum','trimmed_mean':'Trimmed Mean','fltrust':'FLTrust','tasl':'TASL'}
 seen=set()
 for x in data['records']:
  key=(x['algorithm'],x['ratio_percent'],x['seed']);assert key not in seen;seen.add(key)
  p=root/x['source_file'];assert hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
  row=list(csv.DictReader(p.open(encoding='utf-8-sig',newline='')))[x['row']-2]
  assert aliases[row['algo']]==key[0] and round(float(row['byzantine_ratio'])*100)==key[1]
  assert row['dataset']=='cifar10' and row['attack']=='label_flip' and row['alpha']=='0.3'
  for field,metric in [('best_acc','best'),('avg_acc','avg'),('asr','asr')]:
   v=float(row[field]);assert abs(v-targets[key][metric])<.0051
   assert v==x['matched_metrics'][metric]
 missing=set(targets)-seen
 assert len(seen)==59 and missing=={('FLTrust',40,42)}
 result={'validation':'passed','comments':['R4.10','R4.15'],'file_hashes':len(manifest['files']),
         'matched_configurations':len(seen),'metric_checks':len(seen)*3,
         'missing_configurations':data['missing_configurations'],
         'limitations':'Archived rounded summaries; no complete original histories or historical executable/environment certification.'}
 (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()
