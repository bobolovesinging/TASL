"""Offline regeneration from an extracted evidence snapshot; no GPU/network/training.

This verifies recorded results only, not retraining or a public immutable release.
"""
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT/'r4_tasl_experiments_20260921'
FINAL = EXP/'final_protocol_20260928'
OUTPUT = ROOT/'generated'
SEEDS = [42, 123, 2024, 3407, 7711]
T4 = 2.7764451051977987

def read_csv(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def check_files():
    manifest = json.loads((ROOT/'bundle_manifest.json').read_text(encoding='utf-8'))
    for record in manifest['files']:
        path = ROOT/record['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256'], record['path']
    captured = json.loads((FINAL/'results/fashion_single_auditor_five_seed_20261002/reproduction_manifest.json').read_text())
    for record in captured['source_files']:
        assert hashlib.sha256((FINAL/record['path']).read_bytes()).hexdigest()==record['sha256'], record['path']
    principal = EXP/'principal_5seed'
    capture = json.loads((principal/'manifest.json').read_text())
    assert hashlib.sha256((principal/'scripts/exp2/exp2_run.py').read_bytes()).hexdigest()==capture['files']['scripts/exp2/exp2_run.py']
    return len(manifest['files'])

def run_check(script, args):
    result = subprocess.run([sys.executable, str(script), *map(str, args)],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    assert result.returncode == 0, result.stderr
    (OUTPUT/(script.stem+'.txt')).write_text(result.stdout, encoding='utf-8')
    return json.loads(result.stdout)

def workbook_extract():
    data = json.loads((EXP/'legacy_main_sources_20261002/workbook_validation.json').read_text(encoding='utf-8'))
    assert len(data['seed_level_records']) == 60 and len(data['cell_index']) == 130
    for check in data['three_seed_checks']:
        values = check['seed_values']
        assert abs(statistics.mean(values)-check['workbook_mean']) < .011
        assert abs(statistics.stdev(values)-check['workbook_sample_sd']) < .016
    write_csv(OUTPUT/'workbook_seed_metrics.csv', [
        {k: r[k] for k in ['algorithm','ratio_percent','seed','best','avg','asr']}
        for r in data['seed_level_records']])
    source = EXP/'legacy_main_sources_20261002/cifar_workbook_source_records.json'
    matched = 0
    if source.exists():
        historical = json.loads(source.read_text())
        unique = set()
        for record in historical['records']:
            key = (record['algorithm'],record['ratio_percent'],record['seed'])
            assert key not in unique
            unique.add(key)
            target = next(r for r in data['seed_level_records']
                          if (r['algorithm'],r['ratio_percent'],r['seed'])==key)
            assert all(abs(record['matched_values'][k]-target[k]) < .0051 for k in ['best','avg','asr'])
            rounds = record['logged_round_samples']
            assert [r['round'] for r in rounds] == [1,*range(5,201,5)]
            assert len(record['csv_sha256']) == len(record['log_sha256']) == 64
        assert len(unique)==historical['unique_seed_configurations']
        matched=len(unique)
    return {'numeric_values': 130, 'seed_records': 60, 'historical_summary_matches':matched,
        'scope': 'Recompute from the saved, cell-addressed workbook extract. Original workbook was checked read-only before packaging; no unrounded trajectories are implied.'}

def principal_learning(tex):
    root = EXP/'principal_5seed'
    curves = {}
    metrics = {}
    table = []
    summaries = read_csv(root/'aggregate_summary.csv')
    for attack, condition in [('none','clean'), ('label_flip','label_flip_40pct')]:
        for algo in ['fedavg','fltrust','tasl_full']:
            avgs, asrs = [], []
            for seed in SEEDS:
                folder = root/'results'/f'fashion_principal_seed{seed}'
                history = json.loads((folder/f'history__{algo}__{attack}.json').read_text())
                assert history['seed'] == seed and history['algo'] == algo and history['attack'] == attack
                accuracy, asr = history['accuracy'], history['asr']
                assert len(accuracy) == len(asr) == 50
                assert all(math.isfinite(v) for v in accuracy+asr)
                summary = next(r for r in read_csv(folder/'exp2_results.csv')
                               if r['algo'] == algo and r['attack'] == attack)
                avg = statistics.mean(accuracy)
                assert abs(avg-float(summary['avg_acc'])) < 1.e-9
                assert abs(statistics.mean(asr)-float(summary['asr'])) < 1.e-9
                avgs.append(avg); asrs.append(statistics.mean(asr))
                curves[(seed,algo,condition)] = accuracy
            metrics[(algo,condition)] = avgs
            row = {'condition': condition, 'method': algo,
                   'avg_mean': statistics.mean(avgs), 'avg_sample_sd': statistics.stdev(avgs),
                   'asr_mean': statistics.mean(asrs), 'asr_sample_sd': statistics.stdev(asrs)}
            for metric, vals in [('average_acc',avgs), ('lf_asr',asrs)]:
                saved = next(r for r in summaries if r['algo'] == algo and r['condition'] == condition
                             and r['metric'] == metric)
                assert abs(float(saved['mean'])-statistics.mean(vals)) < 1.e-9
                assert abs(float(saved['sd'])-statistics.stdev(vals)) < 1.e-9
            table.append(row)
    block = tex.split('\\label{tab:principal_five_seed}',1)[1].split('\\end{tabular}',1)[0]
    lines = [line for line in block.splitlines() if line.startswith('Clean &') or line.startswith('40\\% LF &')]
    assert len(lines) == 6
    for i, line in enumerate(lines):
        numbers = [float(v) for v in re.findall(r'\d+(?:\.\d+)?', '&'.join(line.split('&')[2:]))]
        expected = [table[i]['avg_mean'],table[i]['avg_sample_sd']]
        if i >= 3: expected += [table[i]['asr_mean'],table[i]['asr_sample_sd']]
        assert len(numbers) == len(expected)
        assert all(abs(a-b) < .0051 for a,b in zip(numbers,expected)), (i,numbers,expected)
    write_csv(OUTPUT/'Table_C1_regenerated.csv',table)
    generated_curves = []
    for algo in ['fedavg','fltrust','tasl_full']:
        for condition in ['clean','label_flip_40pct']:
            for number in range(50):
                values = [curves[(s,algo,condition)][number] for s in SEEDS]
                mean,sd = statistics.mean(values),statistics.stdev(values)
                half = T4*sd/math.sqrt(5)
                generated_curves.append({'algo':algo,'condition':condition,'round':number+1,
                    'mean_accuracy':mean,'sd':sd,'ci95_low':mean-half,'ci95_high':mean+half})
    saved = read_csv(root/'aggregate_curves.csv')
    assert len(saved) == len(generated_curves) == 300
    for a,b in zip(saved,generated_curves):
        for k in ['mean_accuracy','sd','ci95_low','ci95_high']:
            assert abs(float(a[k])-b[k]) < 1.e-9, (a,k)
    write_csv(OUTPUT/'Fig2_curves_regenerated.csv',generated_curves)
    source = (ROOT/'docx_qa/r4_continue_20260929/principal_uncertainty.tex').read_text()
    for condition in ['clean','label_flip_40pct']:
        for algo in ['fedavg','fltrust','tasl_full']:
            rows = [r for r in generated_curves if r['condition'] == condition and r['algo'] == algo]
            for sign in [0,1,-1]:
                coordinates = ' '.join(f"({r['round']},{r['mean_accuracy']+sign*r['sd']:.6f})" for r in rows)
                assert coordinates in source, ('Fig2',condition,algo,sign)
    return {'training_trajectories':30, 'round_records':1500,'Table_C1_numeric_values':18,
            'Fig2_mean_SD_curve_points':300, 'Fig2_TeX_coordinates_checked':True}

def appendix_tables(tex, full, single):
    rows = []
    block = tex.split('\\label{tab:tas_five_seed}',1)[1].split('\\end{tabular}',1)[0]
    lines = [line for line in block.splitlines() if line.startswith(('LF &','Scaling &'))]
    assert len(lines) == 2
    for line, attack in zip(lines,['lf','scaling']):
        result = full['pooled'][attack]
        numbers = [float(v) for v in re.findall(r'\d+(?:\.\d+)?',line.split('&')[1]+' '+line.split('&')[2])]
        expected = [result['avg_accuracy_mean_percent'],result['avg_accuracy_sample_sd_percent'],
                    *result['avg_accuracy_mean_95pct_t_interval']]
        assert all(abs(a-b) < .0051 for a,b in zip(numbers,expected)) and len(numbers)==4
        assert result['blocked']==150 and result['honest_accepted']==100
        rows.append({'table':'A4','attack':attack,'group':'Full TAS/no-Executor',
                    'avg_mean':expected[0],'avg_sd':expected[1], 'recall':100,
                    'paired_delta':0,'delta_ci_low':0,'delta_ci_high':0})
    block = tex.split('\\label{tab:tas_single_five_seed}',1)[1].split('\\end{tabular}',1)[0]
    lines = [line for line in block.splitlines() if line.startswith(('LF &','Scaling &'))]
    assert len(lines)==4
    for line,result in zip(lines,single['pooled']):
        fields = line.split('&')[2:]
        numbers = [float(v) for v in re.findall(r'-?\d+(?:\.\d+)?',' '.join(fields))]
        avg = result['avg_accuracy_seed_stats'];recall = result['recall_seed_stats']
        delta = result['full_minus_single_auditor_paired_avg_accuracy']
        expected = [avg['mean'],avg['sample_sd'],recall['mean'],recall['sample_sd'],
                    delta['mean'],*delta['mean_95pct_t_interval']]
        assert len(numbers)==7 and all(abs(a-b)<.0051 for a,b in zip(numbers,expected)), (numbers,expected)
        rows.append({'table':'A5','attack':result['attack'],'group':result['group'],
                    'avg_mean':avg['mean'],'avg_sd':avg['sample_sd'], 'recall':recall['mean'],
                    'paired_delta':delta['mean'],'delta_ci_low':expected[5],'delta_ci_high':expected[6]})
    write_csv(OUTPUT/'Tables_A4_A5_regenerated.csv',rows)
    return {'Full_reference_rounds':1000, 'single_auditor_rounds':1000,
            'matched_initial_conditions':sum(r['initial_parent_matches_reference'] and r['first_round_update_hashes_match_reference'] for r in single['cases']),
            'paired_Avg_CIs_include_zero':all(r['full_minus_single_auditor_paired_avg_accuracy']['mean_95pct_t_interval'][0] <= 0 <= r['full_minus_single_auditor_paired_avg_accuracy']['mean_95pct_t_interval'][1] for r in single['pooled'])}

def main():
    checked = check_files()
    OUTPUT.mkdir(exist_ok=True)
    tex = (ROOT/'manuscript/main.tex').read_text(encoding='utf-8')
    legacy = run_check(ROOT/'docx_qa/validate_legacy_sources_20261002.py',[])
    full = run_check(FINAL/'validate_fashion_fixed_five_seed.py',[FINAL/'results/fashion_fixed_five_seed_20260930'])
    single_root = FINAL/'results/fashion_single_auditor_five_seed_20261002'
    run_check(FINAL/'validate_single_auditor_five_seed.py',[single_root])
    single = json.loads((single_root/'validation.json').read_text())
    recent=run_check(ROOT/'docx_qa/validate_recent_baseline_20261002.py',[])
    sweep=run_check(ROOT/'docx_qa/validate_trust_sweep_20261002.py',[])
    baseline_block=tex.split('\\label{tab:recent_baseline}',1)[1].split('\\end{tabular}',1)[0]
    for name,label in [('fedavg','FedAvg'),('lasa','LASA (author defaults)'),('tasl_full','TASL')]:
        line=next(line for line in baseline_block.splitlines() if line.startswith(label+' &'))
        values=[float(v) for v in re.findall(r'\d+(?:\.\d+)?','&'.join(line.split('&')[1:]))]
        clean=next(r for r in recent['rows'] if r['algorithm']==name and r['attack']=='none')
        attacked=next(r for r in recent['rows'] if r['algorithm']==name and r['attack']=='label_flip')
        assert len(values)==3 and all(abs(a-b)<.0051 for a,b in zip(values,[clean['avg_acc'],attacked['avg_acc'],attacked['avg_asr']]))
    sweep_block=tex.split('\\label{tab:fixed_constant_sensitivity}',1)[1].split('\\end{tabular}',1)[0]
    labels={'trust_power':'Cosine power','norm_penalty':'Norm penalty','weight_cap_ratio':'Weight-cap ratio',
            'anchor_current_fraction':'Current-anchor fraction','norm_gate_multiplier':'Norm-gate multiplier'}
    for row in sweep['rows']:
        candidates=[line for line in sweep_block.splitlines() if line.startswith(labels[row['parameter']]+' &')]
        line=next(line for line in candidates if float(line.split('&')[1])==row['value'])
        values=[float(v) for v in re.findall(r'\d+(?:\.\d+)?','&'.join(line.split('&')[2:]))]
        assert len(values)==3 and all(abs(a-b)<.0051 for a,b in zip(values,[row['avg_acc'],row['avg_asr'],row['honest_zero_percent']]))
    result = {'validation':'passed','comments':['R4.10','R4.15'], 'snapshot_files_sha256_checked':checked,
        'historical_governance':legacy['matched_cells'], 'workbook_extract':workbook_extract(),
        'principal_learning':principal_learning(tex),'five_seed_governance':appendix_tables(tex,full,single),
        'recent_baseline':{'trajectories':recent['trajectories'],'rounds':recent['rounds'],'table_numeric_entries':9},
        'fixed_constant_sensitivity':{'trajectories':sweep['trajectories'],'rounds':sweep['round_records'],'scan_numeric_entries':30},
        'limitations':['Offline statistical regeneration is not retraining.',
            'Unrecovered original main-table trajectories and historical executable versions remain missing.',
            'Only this selected subset is packaged; all-table/all-figure regeneration and immutable public release remain pending.',
            'Captured package versions are not a cryptographic dependency lock or a proof of identical training outputs on another machine.']}
    (OUTPUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
