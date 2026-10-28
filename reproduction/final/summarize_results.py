"""Regenerate five-seed point summaries and paired, exploratory t intervals."""
from pathlib import Path
import json,numpy as np
ROOT=Path(__file__).resolve().parent
def summarize():
 out={}
 for attack in ['none','label_flip']:
  for method in ['fedavg','fltrust','tasl_full','lasa']:
   rows=[json.loads(x.read_text()) for x in sorted((ROOT/'results/c6_repeated').glob(f'{method}_{attack}_seed*/summary.json'))]
   assert len(rows)==5
   out[f'{method}_{attack}']={k:dict(mean=float(np.mean([x[k] for x in rows])),sd=float(np.std([x[k] for x in rows],ddof=1))) for k in ['avg','final']+(['lf_asr'] if attack=='label_flip' else [])}
   for x in rows:
    data=json.loads((ROOT/'results/c6_repeated'/f"{method}_{attack}_seed{x['seed']}"/'metrics.json').read_text())
    assert len(data['round_metrics'])==50
  for baseline in ['fedavg','fltrust','lasa']:
   a={x['seed']:x['avg'] for x in [json.loads(q.read_text()) for q in (ROOT/'results/c6_repeated').glob(f'tasl_full_{attack}_seed*/summary.json')]}
   b={x['seed']:x['avg'] for x in [json.loads(q.read_text()) for q in (ROOT/'results/c6_repeated').glob(f'{baseline}_{attack}_seed*/summary.json')]}
   v=np.array([a[s]-b[s] for s in sorted(a)]);margin=2.7764451051977987*v.std(ddof=1)/np.sqrt(5)
   out[f'paired_{attack}_{baseline}']=dict(mean=float(v.mean()),ci=[float(v.mean()-margin),float(v.mean()+margin)],n=5,df=4,multiple_comparison_adjustment=False)
 return out
if __name__=='__main__':
 out=summarize();(ROOT/'results/c6_statistics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
