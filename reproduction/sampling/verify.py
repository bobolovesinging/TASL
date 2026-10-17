"""Regenerate selected recorded evidence without GPU, network or third-party packages."""
import csv, hashlib, itertools, json, math, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def read(name):
    with (ROOT/name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def load(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))

def quantile(values, q=.1):
    a=sorted(values); h=(len(a)-1)*q; lo=math.floor(h)
    return a[lo]+(h-lo)*(a[min(lo+1,len(a)-1)]-a[lo])

def main():
    manifest=load('manifest.json')
    for item in manifest['files']:
        p=(ROOT/item['path']).resolve()
        assert ROOT.resolve() in p.parents
        assert p.stat().st_size==item['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'], item['path']
    sources=load('cifar_sources/matched_records.json')['records']
    authority=load('cifar_sources/workbook_seed_records.json')
    targets={(x['algorithm'],x['ratio_percent'],x['seed']):x for x in authority['seed_level_records']}
    aliases={'fedavg':'FedAvg','multi_krum':'Multi-Krum','trimmed_mean':'Trimmed Mean','fltrust':'FLTrust','tasl':'TASL'}
    seen=set(); counts={'archived_csv':0,'author_workbook':0}
    for item in sources:
        key=(item['algorithm'],item['ratio_percent'],item['seed']); assert key not in seen
        seen.add(key); counts[item['origin']]+=1
        blob=(ROOT/'cifar_sources'/item['source_file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==item['sha256']
        row=read('cifar_sources/'+item['source_file'])[item['row']-2]
        assert aliases[row['algo']]==key[0] and round(float(row['byzantine_ratio'])*100)==key[1]
        assert row['dataset']=='cifar10' and row['attack']=='label_flip' and float(row['alpha'])==.3
        for field,metric in [('best_acc','best'),('avg_acc','avg'),('asr','asr')]:
            assert abs(float(row[field])-targets[key][metric])<.0051
            assert float(row[field])==item['matched_metrics'][metric]
        if item['origin']=='author_workbook':
            assert key==('FLTrust',40,42) and row['last_acc']==''
            assert item['source_cells']==targets[key]['source_cells']
    assert seen==set(targets) and len(seen)==60 and counts=={'archived_csv':59,'author_workbook':1}

    captures=load('sampling/captured_weights.json')['rounds']
    expected={}; out=[]
    for capture in captures:
        rnd=capture['round']; fresh={int(k):v for k,v in capture['weights'].items()}
        assert set(fresh)==set(range(10))
        order=sorted(fresh,key=lambda i:(fresh[i],i))
        for m in [2,4,6]:
            proposed=fresh.copy()
            for recipient,donor in zip(order[:m//2],reversed(order[-m//2:])):
                assert fresh[donor]>.01
                proposed[recipient]+=.01; proposed[donor]-=.01
            altered={i for i in fresh if abs(proposed[i]-fresh[i])>1e-6}
            assert len(altered)==m and min(proposed.values())>=0
            assert abs(sum(proposed.values())-sum(fresh.values()))<1e-12
            for k in range(1,11):
                expected[(rnd,m,k)]=set(itertools.combinations(range(10),k))
    detected={key:0 for key in expected}; totals={key:0 for key in expected}
    raw=read('sampling/subsets.csv')
    for row in raw:
        key=tuple(int(row[x]) for x in ['round','changed_weights','sampled_clients'])
        subset=tuple(map(int,row['sample_ids'].split('|')))
        assert subset in expected[key]; expected[key].remove(subset)
        capture=next(x for x in captures if x['round']==key[0]); fresh={int(k):v for k,v in capture['weights'].items()}
        order=sorted(fresh,key=lambda i:(fresh[i],i)); m=key[1]
        altered=set(order[:m//2]+order[-m//2:])
        assert set(map(int,row['changed_ids'].split('|')))==altered
        hit=int(bool(altered.intersection(subset)))
        assert int(row['vp_detected'])==hit and int(row['vs_detected'])==1
        assert int(row['vp_honest_rejected'])==0 and float(row['sample_ratio'])==key[2]/10
        detected[key]+=hit; totals[key]+=1
    assert len(raw)==9207 and all(not x for x in expected.values())
    summaries=read('sampling/summary.csv'); assert len(summaries)==90
    for row in summaries:
        key=tuple(int(row[x]) for x in ['round','changed_weights','sampled_clients']); _,m,k=key
        probability=1-(math.comb(10-m,k)/math.comb(10,k) if k<=10-m else 0)
        assert int(row['detected'])==detected[key] and int(row['subsets'])==totals[key]
        assert abs(float(row['vp_weight_detection_pct'])-100*probability)<1e-10
        assert abs(float(row['theoretical_pct'])-100*probability)<1e-10
        assert float(row['honest_acceptance_pct'])==100 and float(row['false_rejection_pct'])==0
        assert float(row['vs_weight_detection_pct'])==100
        if key[0]==1: out.append({'sampled_clients':k,'altered_weights':m,'detection_percent':100*probability})
    timings=read('sampling/timings.csv');assert len(timings)==6030
    assert all(math.isfinite(float(r['weight_check_ms'])) and float(r['weight_check_ms'])>=0 for r in timings)
    for key in itertools.product([1,2,3],range(1,11)):
        group=[r for r in timings if (int(r['round']),int(r['sampled_clients']))==key]
        assert len(group)==201 and {int(r['repeat']) for r in group}==set(range(201))

    measured=read('storage/measured_round_bytes.csv');assert len(measured)==3
    u,c,g=[statistics.mean(float(x[k]) for x in measured) for k in ['client_updates_bytes','checkpoint_bytes','permanent_metadata_bytes']]
    accounting=read('storage/storage_accounting.csv');assert len(accounting)==48
    for row in accounting:
        t,m,q=map(int,[row['rounds'],row['window'],row['archive_replication_factor']]);full=t*(u+c+g)
        if row['scheme']=='full_retention':active,archive=full,0
        else:active=min(m,t)*u+c+t*g;archive=max(t-m,0)*u+max(t-1,0)*c
        for k,v in [('active_bytes',active),('archive_bytes_single_copy',archive),('replicated_archive_bytes',archive*q),('total_physical_bytes',active+archive*q)]:
            assert int(row[k])==round(v)
    storage=load('storage/semantic_retention_summary.json');t,m=storage['rounds'],storage['window']
    full=t*(u+c+g); uniform=m*u+c+m*g;hbs=m*u+c+t*g
    assert [p['active_bytes'] for p in storage['policies']]==[full,uniform,hbs,hbs]
    assert hbs-uniform==64350
    assert abs(storage['hbs_extra_percent_vs_uniform']-100*(hbs-uniform)/uniform)<1e-12

    metric_rows=read('learning/per_seed_metrics.csv');histories=0; p10={}
    for seed in [42,123,2024,3407,7711]:
        for algo in ['fedavg','fltrust','tasl_full']:
            for attack,condition in [('none','clean'),('label_flip','label_flip_40pct')]:
                history=load(f'learning/histories/seed{seed}/history__{algo}__{attack}.json')
                assert (history['seed'],history['algo'],history['attack'])==(seed,algo,attack)
                assert len(history['accuracy'])==50 and all(math.isfinite(v) for v in history['accuracy'])
                value=quantile(history['accuracy']);p10[(seed,algo,condition)]=value;histories+=1
                row=next(x for x in metric_rows if (int(x['seed']),x['algo'],x['condition'])==(seed,algo,condition))
                assert abs(value-float(row['p10_acc']))<1e-10
    differences=[p10[(s,'tasl_full','label_flip_40pct')]-p10[(s,'fltrust','label_flip_40pct')] for s in [42,123,2024,3407,7711]]
    lo,hi=0.,1.
    for _ in range(100):
        y=(lo+hi)/2
        if .5+.75*y-.25*y**3<.975:lo=y
        else:hi=y
    y=(lo+hi)/2;t4=2*y/math.sqrt(1-y*y)
    mean=statistics.mean(differences);half=t4*statistics.stdev(differences)/math.sqrt(5)
    existing=load('learning/lower_tail_verification.json')
    for k,v in [('mean_difference',mean),('ci95_low',mean-half),('ci95_high',mean+half)]:assert abs(existing[k]-v)<1e-10
    saved=next(r for r in read('learning/paired_effects.csv') if r['condition']=='label_flip_40pct' and r['comparison']=='tasl_full-minus-fltrust' and r['metric']=='p10_acc')
    assert abs(float(saved['mean_difference'])-mean)<1e-10 and abs(float(saved['ci95_low'])-(mean-half))<1e-10 and abs(float(saved['ci95_high'])-(mean+half))<1e-10

    tex=(ROOT/'table_excerpts.tex').read_text(encoding='utf-8')
    for value in [full,uniform,hbs]:assert f'{int(value):,}' in tex
    block=tex.split(r'\label{tab:vp_sampling_sensitivity}',1)[1]
    for k in range(1,11):
        values=[100*(1-(math.comb(10-m,k)/math.comb(10,k) if k<=10-m else 0)) for m in [2,4,6]]
        line=f'{k}/10 & {k*10} & '+' & '.join(f'{v:.2f}' for v in values)+r' \\'
        assert line in block
    generated=ROOT/'generated';generated.mkdir(exist_ok=True)
    with (generated/'Table_D7_regenerated.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(out[0]));writer.writeheader();writer.writerows(out)
    result={'validation':'passed','file_hashes':len(manifest['files']),
            'cifar_lf':{'configurations':60,'metrics':180,'origins':counts},
            'sampling':{'raw_cases':9207,'settings':90,'recorded_timings_checked':6030,'table_D7_rows':10},
            'storage':{'archive_accounting_rows':48,'table7_policies':4,'hbs_extra_bytes':64350},
            'lower_tail':{'histories':histories,'rounds':histories*50,'mean_difference':mean,'ci95':[mean-half,mean+half],'secondary_exploratory_no_multiplicity_adjustment':True},
            'scope':'Recorded-result regeneration only; no retraining, new runtime measurement, full historical provenance or deployed system claim.'}
    (generated/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
