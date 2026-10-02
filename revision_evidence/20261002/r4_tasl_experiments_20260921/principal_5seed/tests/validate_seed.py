from pathlib import Path
import csv, json, math, sys
result_dir = Path(sys.argv[1])
expected = {(a,b) for a in ['fedavg','fltrust','tasl_full'] for b in ['none','label_flip']}
rows = list(csv.DictReader(open(result_dir/'exp2_results.csv', encoding='utf-8')))
assert len(rows) == 6, (result_dir, len(rows))
assert {(r['algo'],r['attack']) for r in rows} == expected
for algo, attack in expected:
    hist = json.load(open(result_dir/f'history__{algo}__{attack}.json', encoding='utf-8'))
    assert len(hist['accuracy']) == 50
    assert len(hist['asr']) == 50
    assert all(math.isfinite(float(x)) for x in hist['accuracy'] + hist['asr'])
    metrics = list(csv.DictReader(open(result_dir/f'round_metrics__{algo}__{attack}.csv', encoding='utf-8')))
    assert len(metrics) == 50
    if algo == 'tasl_full':
        for row in metrics:
            weights = json.loads(row['weights_json'])
            assert len(weights) == 10
            assert abs(sum(float(x) for x in weights.values()) - 1.0) < 1e-5
print('SEED_VALIDATION_PASS', result_dir.name)
