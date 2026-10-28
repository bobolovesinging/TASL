"""Export central final-method numerical tables directly from released records."""
from pathlib import Path
import json,statistics
BASE=Path(__file__).resolve().parent;RECORDS=BASE/'records'
def read(name):return json.loads((RECORDS/name).read_text())
def tex(caption,columns,rows):
 return '\\paragraph{'+caption+'}\n\\begin{tabular}{'+'l'*len(columns)+'}\n'+' & '.join(columns)+r' \\'+'\n'+'\n'.join(' & '.join(str(c).replace('%',r'\%') for c in row)+r' \\' for row in rows)+'\n\\end{tabular}\n'
blocks=[];learning=[]
for attack,condition in [('none','Clean'),('label_flip','40% LF')]:
 for method,label in [('fedavg','FedAvg'),('fltrust','FLTrust'),('tasl_full','TASL'),('lasa','LASA')]:
  summaries=[read(f'c6_repeated/{method}_{attack}_seed{s}/summary.json') for s in [42,123,2024,3407,7711]]
  pm=lambda key:f"{statistics.mean(x[key] for x in summaries):.2f} $\\pm$ {statistics.stdev(x[key] for x in summaries):.2f}"
  learning.append([condition,label,pm('avg'),pm('final'),pm('lf_asr') if attack=='label_flip' else '--'])
blocks.append(tex('Five-seed learning',['Condition','Method','Avg','Final','LF ASR'],learning))
blocks.append(tex('Fixed-rule exclusion caps',['Condition','Cap','Avg','Final','Honest zero','Honest out','Byzantine out'],
 [[x['condition'],x['budget'],f"{x['avg_accuracy']:.2f}",f"{x['final_accuracy']:.2f}",f"{x['honest_zero_opportunities']}/{x['honest_opportunities']}",x['honest_excluded'],x['byzantine_excluded']] for x in read('c3_analysis_v2/budget_summary.json')]))
rows=[]
for name in ['reference','trust_only','full_tas']:
 d=read(f'c2/{name}/summary.json');metrics=read(f'c2/{name}/round_metrics.json')
 rows.append([name,d['true_rejections'],d['false_rejections'],f"{d['average_accuracy']:.2f}",f"{metrics[-1]['accuracy']:.2f}"])
blocks.append(tex('Paired malicious Executor',['Group','True rejects','False rejects','Avg','Final'],rows))
blocks.append(tex('Complete-verifier controls',['Design','Verify ms','Recover ms','Logical input MiB'],[[name,f"{d['verify_median_ms']:.2f}",f"{d['recovery_median_ms']:.2f}",f"{d['logical_received_MiB_median']:.2f}"] for name,d in read('c4/summary.json')['configurations'].items()]))
blocks.append(tex('Final retained bytes',['Policy','Active MiB','Archive MiB','Reduction','Historical queries'],[[name,f"{d['active_bytes']/2**20:,.3f}",f"{d['archive_bytes']/2**20:,.3f}",f"{d['active_reduction_pct']:.3f}",f"{d['historical_records_without_archive']}/45"] for name,d in read('hbs_final_v2/summary.json')['policies'].items()]))
output=BASE/'analysis';output.mkdir(exist_ok=True);(output/'central_tables.tex').write_text('\n'.join(blocks));print(output/'central_tables.tex')
