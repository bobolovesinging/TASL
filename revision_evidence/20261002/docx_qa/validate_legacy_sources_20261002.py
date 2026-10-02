"""Recompute submitted table cells from recovered historical round CSVs."""
import csv,hashlib,json,re,statistics
from pathlib import Path
root=Path(__file__).resolve().parents[1]
source=root/'r4_tasl_experiments_20260921/legacy_sources_20261002'
tex=(root/'manuscript/main.tex').read_text(encoding='utf-8')
groups=['fedavg','trust_only','trust_v1','trust_v2','trust_v1v2']
conditions=['mnist_lf','mnist_scaling','fashion_lf','fashion_scaling']
def rows(path):
 with path.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
cells=[];cases=[];table_values={}
for label,folder,byz,table in [('tab:governance','table10_noexcl',3,4),('tab:governance_40','table11_noexcl',4,5)]:
 block=tex.split('\\label{'+label+'}',1)[1].split('\\end{table*}',1)[0]
 lines=[line for line in block.splitlines() if '&' in line and line.rstrip().endswith('\\\\') and ('No Defense' in line or 'Trust Fusion Only' in line or line.startswith('\\revRfour{Trust +') or line.startswith('\\revRfour{\\textbf{Full TAS}'))]
 assert len(lines)==5
 numerical=[[float(re.search(r'\d+(?:\.\d+)?',part).group()) for part in line.split('&')[1:]] for line in lines]
 table_values[table]=numerical
 assert all(len(values)==12 for values in numerical)
 for ci,condition in enumerate(conditions):
  directory=source/folder/condition
  raw=directory/f'tasv2_mixed12_r50_b{byz}.csv';summary=directory/'tasv2_summary.csv'
  data=rows(raw);reported={r['group']:r for r in rows(summary)}
  assert len(data)==300 and len(reported)==6
  for gi,group in enumerate(groups):
   series={int(r['round']):r for r in data if r['group']==group}
   assert sorted(series)==list(range(1,51))
   attack=[r for r in series.values() if int(r['attack_active'])]
   assert len(attack)==30
   avg=statistics.mean(float(r['accuracy']) for r in series.values())
   blocked=sum(not int(r['consensus_passed']) for r in attack)
   rate=100*blocked/30
   assert abs(avg-float(reported[group]['avg_acc']))<.000051
   assert blocked==int(reported[group]['blocked_rounds'])
   for offset,metric,value in [(0,'Avg',avg),(1,'Interc',rate)]:
    submitted=numerical[gi][ci*2+offset]
    assert abs(value-submitted)<.051,(table,condition,group,metric,value,submitted)
    cells.append({'table':table,'condition':condition,'group':group,'metric':metric,'manuscript_value':submitted,'recomputed_value':value,'raw_csv':str(raw.relative_to(root)),'summary_csv':str(summary.relative_to(root))})
  cases.append({'table':table,'condition':condition,'round_rows':len(data),'csv_sha256':sha(raw),'summary_sha256':sha(summary)})
for table,condition,filename,ci in [(4,'cifar_lf','gov_cifar10_r30_final.log',4),(5,'cifar_lf','gov_cifar10_r40_final.log',4),(4,'cifar_scaling','gov_cifar10_scaling_full_r30.log',5)]:
 path=source/'governance_ablation'/filename
 text=path.read_text(encoding='utf-8')
 pattern=r'^\[(\d+)\]\s+(fedavg|trust_only|v1_only|v2_only|tas)\s+\|\s*atk=\s*(\w+).*?consensus=(\d).*?acc=([\d.]+)%'
 records=re.findall(pattern,text,re.M)
 assert len(records)==1000,(path,len(records))
 kinds=set()
 for gi,legacy in enumerate(['fedavg','trust_only','v1_only','v2_only','tas']):
  series={int(n):(atk,int(accepted),float(acc)) for n,g,atk,accepted,acc in records if g==legacy}
  assert sorted(series)==list(range(1,201))
  attacked=[r for r in series.values() if r[0]!='None']
  assert len(attacked)==120,(path,legacy,len(attacked))
  kinds.update(r[0] for r in attacked)
  avg=statistics.mean(r[2] for r in series.values())
  rate=100*sum(not r[1] for r in attacked)/120
  for offset,metric,value in [(0,'Avg',avg),(1,'Interc',rate)]:
   submitted=table_values[table][gi][ci*2+offset]
   # Per-round logs and their end summaries were rounded to 0.01 point;
   # manuscript one-decimal cells may reflect that earlier rounding.
   allowance=.011 if table==4 and ci==5 and metric=='Avg' else (.055 if metric=='Avg' else .051)
   assert abs(value-submitted)<allowance,(path,legacy,metric,value,submitted)
   cells.append({'table':table,'condition':condition,'group':legacy,'metric':metric,'manuscript_value':submitted,'recomputed_value':value,'raw_log':str(path.relative_to(root)),'accuracy_precision':'Per-round accuracy printed to 0.01 percentage point; full-precision tensors not recovered.'})
 cases.append({'table':table,'condition':condition,'round_rows':1000,'log_sha256':sha(path),'legacy_attack_types':sorted(kinds)})
for gpu,mapping in [('gpu0',[(1,'trust_only'),(2,'v1')]),('gpu1',[(3,'v2'),(4,'tas')])]:
 path=source/'cifar_scaling_20260730'/gpu/'cifar_scaling_governance_seed42.json'
 obj=json.loads(path.read_text(encoding='utf-8'))
 for gi,group in mapping:
  item=obj['results'][group];history=item['acc_history'];attacks=item['executor_history']
  assert len(history)==200 and len(attacks)==120
  assert len({r['round'] for r in attacks})==120
  avg=statistics.mean(history);rate=100*sum(r['blocked'] for r in attacks)/120
  assert abs(avg-item['avg_acc'])<1.e-10
  for offset,metric,value in [(0,'Avg',avg),(1,'Interc',rate)]:
   submitted=table_values[5][gi][10+offset]
   assert abs(value-submitted)<.051
   cells.append({'table':5,'condition':'cifar_scaling','group':group,'metric':metric,'manuscript_value':submitted,'recomputed_value':value,'raw_json':str(path.relative_to(root)),'source_precision':'Full-precision 200-round accuracy history and 120 attacked-round records.'})
 cases.append({'table':5,'condition':'cifar_scaling','groups':[g for _,g in mapping],'json_sha256':sha(path)})
path=source/'governance_ablation/gov_cifar10_scaling_full_r40.log'
text=path.read_text(encoding='utf-8')
records=re.findall(pattern,text,re.M)
series={int(n):(atk,int(accepted),float(acc)) for n,g,atk,accepted,acc in records if g=='fedavg'}
assert sorted(series)==list(range(1,201))
attacked=[r for r in series.values() if r[0]!='None'];assert len(attacked)==120
for offset,metric,value in [(0,'Avg',statistics.mean(r[2] for r in series.values())),(1,'Interc',100*sum(not r[1] for r in attacked)/120)]:
 submitted=table_values[5][0][10+offset];assert abs(value-submitted)<.055
 cells.append({'table':5,'condition':'cifar_scaling','group':'fedavg','metric':metric,'manuscript_value':submitted,'recomputed_value':value,'raw_log':str(path.relative_to(root)),'source_precision':'Rounded per-round log; different historical script from the four audit rows.'})
cases.append({'table':5,'condition':'cifar_scaling','groups':['fedavg'],'log_sha256':sha(path)})
assert len(cells)==120
result={'validation':'passed','comments':['R4.10','R4.15'],'matched_cells':len(cells),'cases':cases,'cell_index':cells,
 'limitations':'Historical executable commits/environments not certified. Some CIFAR cells are compatible with rounded logs, not full-precision CSV recovery. CIFAR Table 5 scaling uses July JSON audit rows and a June FedAvg log, rather than one uniform source file. Main Tables 2/3 still require additional provenance. Legacy honest-round acceptance is not a measured false-positive audit test. CIFAR legacy LF logs contain A3 joint tampering as well as A1/A2, unlike the final A1/A2 schedule.'}
(source/'validation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({'validation':'passed','matched_cells':len(cells),'cases':len(cases)},ensure_ascii=False))
