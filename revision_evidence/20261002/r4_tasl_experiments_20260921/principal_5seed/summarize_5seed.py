from pathlib import Path
import csv, json, math, sys
import numpy as np
from scipy.stats import t

root = Path(sys.argv[1])
seeds = [42,123,2024,3407,7711]
algos = ['fedavg','fltrust','tasl_full']
attacks = ['none','label_flip']
records=[]
curves={}
for seed in seeds:
    d=root/'results'/f'fashion_principal_seed{seed}'
    rows={(r['algo'],r['attack']):r for r in csv.DictReader(open(d/'exp2_results.csv',encoding='utf-8'))}
    assert set(rows)=={(a,b) for a in algos for b in attacks}
    for algo in algos:
        for attack in attacks:
            r=rows[(algo,attack)]
            h=json.load(open(d/f'history__{algo}__{attack}.json',encoding='utf-8'))
            acc=np.asarray(h['accuracy'],dtype=float); asr_curve=np.asarray(h['asr'],dtype=float)
            assert len(acc)==50 and np.isfinite(acc).all() and np.isfinite(asr_curve).all()
            curves[(seed,algo,attack)]=acc
            records.append({
                'seed':seed,'algo':algo,'condition':'clean' if attack=='none' else 'label_flip_40pct',
                'actual_byzantine_ratio':0.0 if attack=='none' else 0.4,
                'configured_tolerance':0.4,
                'best_acc':float(r['best_acc']),'average_acc':float(r['avg_acc']),'final_acc':float(r['last_acc']),
                'lf_asr':float(r['asr']),
                'aulc':float(np.trapz(acc,dx=1)/(len(acc)-1)),
                'p10_acc':float(np.percentile(acc,10)),
                'round_to_95pct_best':int(np.argmax(acc >= 0.95*np.max(acc))+1),
            })

fields=list(records[0])
with open(root/'per_seed_metrics.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(records)

def ci(vals):
    x=np.asarray(vals,dtype=float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1))
    half=float(t.ppf(0.975,n-1)*sd/math.sqrt(n)) if n>1 else float('nan')
    return mean,sd,mean-half,mean+half
summary=[]
metric_names=['best_acc','average_acc','final_acc','lf_asr','aulc','p10_acc','round_to_95pct_best']
for algo in algos:
    for condition in ['clean','label_flip_40pct']:
        subset=[r for r in records if r['algo']==algo and r['condition']==condition]
        for metric in metric_names:
            vals=[r[metric] for r in subset]
            mean,sd,lo,hi=ci(vals)
            summary.append({'algo':algo,'condition':condition,'metric':metric,'n':len(vals),'mean':mean,'sd':sd,'ci95_low':lo,'ci95_high':hi})
with open(root/'aggregate_summary.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)

# Paired method effects; positive accuracy differences favor TASL, negative ASR differences favor TASL.
paired=[]
for condition in ['clean','label_flip_40pct']:
    for comparator in ['fedavg','fltrust']:
        for metric in metric_names:
            diffs=[]
            for seed in seeds:
                ta=next(r for r in records if r['seed']==seed and r['algo']=='tasl_full' and r['condition']==condition)
                co=next(r for r in records if r['seed']==seed and r['algo']==comparator and r['condition']==condition)
                diffs.append(float(ta[metric])-float(co[metric]))
            mean,sd,lo,hi=ci(diffs); dz=mean/sd if sd>0 else float('inf')
            paired.append({'condition':condition,'comparison':f'tasl_full-minus-{comparator}','metric':metric,'n':len(diffs),'mean_difference':mean,'sd_difference':sd,'ci95_low':lo,'ci95_high':hi,'cohen_dz':dz})
with open(root/'paired_effects.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(paired[0])); w.writeheader(); w.writerows(paired)

# Within-method degradation from clean to attack; positive means attack lowers the metric.
degradation=[]
for algo in algos:
    for metric in ['best_acc','average_acc','final_acc','aulc','p10_acc']:
        vals=[]
        for seed in seeds:
            clean=next(r for r in records if r['seed']==seed and r['algo']==algo and r['condition']=='clean')
            atk=next(r for r in records if r['seed']==seed and r['algo']==algo and r['condition']=='label_flip_40pct')
            vals.append(float(clean[metric])-float(atk[metric]))
        mean,sd,lo,hi=ci(vals)
        degradation.append({'algo':algo,'metric':metric,'n':len(vals),'clean_minus_attack':mean,'sd':sd,'ci95_low':lo,'ci95_high':hi})
with open(root/'robustness_degradation.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(degradation[0])); w.writeheader(); w.writerows(degradation)

# TASL decision diagnostics.
diag=[]
for seed in seeds:
    for attack in attacks:
        d=root/'results'/f'fashion_principal_seed{seed}'
        rows=list(csv.DictReader(open(d/f'round_metrics__tasl_full__{attack}.csv',encoding='utf-8')))
        assert len(rows)==50
        honest_n=10 if attack=='none' else 6
        byz_n=0 if attack=='none' else 4
        honest_rej=sum(int(r['honest_rejected']) for r in rows)
        byz_nonzero=sum(int(r['byzantine_nonzero']) for r in rows) if byz_n else 0
        final_excluded=int(rows[-1]['excluded_total'])
        all_excluded_round=''
        if byz_n:
            for r in rows:
                if int(r['excluded_total'])>=byz_n:
                    all_excluded_round=int(r['round']); break
        diag.append({
            'seed':seed,'condition':'clean' if attack=='none' else 'label_flip_40pct',
            'honest_client_round_decisions':honest_n*50,
            'honest_zero_weight_decisions':honest_rej,
            'honest_zero_weight_rate':honest_rej/(honest_n*50),
            'byzantine_client_round_decisions':byz_n*50,
            'byzantine_nonzero_weight_decisions':byz_nonzero,
            'byzantine_nonzero_weight_rate':byz_nonzero/(byz_n*50) if byz_n else '',
            'mean_byzantine_weight_share':float(np.mean([float(r['byzantine_weight_sum']) for r in rows])) if byz_n else '',
            'final_excluded_clients':final_excluded,
            'round_all_four_excluded':all_excluded_round,
        })
with open(root/'tasl_weight_diagnostics.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(diag[0])); w.writeheader(); w.writerows(diag)

# Aggregate curve means and pointwise t intervals.
curve_rows=[]
for algo in algos:
    for attack in attacks:
        mat=np.stack([curves[(s,algo,attack)] for s in seeds])
        for idx in range(50):
            mean,sd,lo,hi=ci(mat[:,idx])
            curve_rows.append({'algo':algo,'condition':'clean' if attack=='none' else 'label_flip_40pct','round':idx+1,'mean_accuracy':mean,'sd':sd,'ci95_low':lo,'ci95_high':hi})
with open(root/'aggregate_curves.csv','w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(curve_rows[0])); w.writeheader(); w.writerows(curve_rows)

validation={
 'seeds':seeds,'training_combinations':len(records),'round_records':len(records)*50,
 'expected_training_combinations':30,'expected_round_records':1500,
 'all_finite':all(math.isfinite(float(r[m])) for r in records for m in metric_names),
 'complete_marker':(root/'COMPLETE').exists(),
 'source_sha256':json.load(open(root/'manifest.json'))['files']['scripts/exp2/exp2_run.py'],
}
assert validation['training_combinations']==30 and validation['round_records']==1500 and validation['all_finite'] and validation['complete_marker']
(root/'validation_report.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print('CROSS_SEED_VALIDATION_PASS',json.dumps(validation))
for row in summary:
    if row['metric'] in ('average_acc','final_acc','lf_asr'):
        print('SUMMARY',row)
print('DIAGNOSTICS')
for row in diag: print(row)
