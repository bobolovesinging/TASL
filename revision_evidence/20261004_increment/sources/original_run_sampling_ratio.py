"""Finite-subset validation of the frozen VP weight cross-check; no retraining."""
from pathlib import Path
from typing import Dict, List, Tuple, Set
import ast, csv, hashlib, itertools, json, math, time
import numpy as np

E = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SOURCE = E/'final_protocol_20260928/scripts/exp3/exp3_governance_tasv2.py'
source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
assert source_hash == 'f71067a01c7af357c664306aed944a64c440e53df537a0f2f7e8f0311311e987'
tree = ast.parse(SOURCE.read_text(encoding='utf-8-sig'))
names = {'v1_verify_trust', 'v2_verify_trust'}
nodes = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in names]
assert len(nodes) == 2
env = dict(np=np, Dict=Dict, List=List, Tuple=Tuple, Set=Set, TOLERANCE=1e-6)
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)
vs, vp = env['v1_verify_trust'], env['v2_verify_trust']
real_rng = np.random.RandomState
rows, summaries, timings, captures = [], [], [], []

class FixedSubset:
    def __init__(self, subset): self.subset = subset
    def choice(self, n, size, replace=False):
        assert n == 10 and size == len(self.subset) and replace is False
        return np.array(self.subset)

for capture in sorted((E/'audit_replay/audit_inputs').glob('round_*.npz')):
    z = np.load(capture)
    ids = z['client_ids'].tolist()
    assert ids == list(range(10))
    fresh = dict(zip(ids, map(float, z['weights'])))
    order = sorted(ids, key=lambda i: (fresh[i], i))
    captures.append(dict(file=str(capture.relative_to(E)), sha256=hashlib.sha256(capture.read_bytes()).hexdigest()))
    round_id = int(z['round'][0])
    for m in [2, 4, 6]:
        published = fresh.copy()
        for recipient, donor in zip(order[:m//2], reversed(order[-m//2:])):
            assert fresh[donor] > 0.01
            published[recipient] += 0.01
            published[donor] -= 0.01
        altered = {i for i in ids if abs(published[i]-fresh[i]) > 1e-6}
        assert len(altered) == m and min(published.values()) >= 0
        assert abs(sum(published.values())-sum(fresh.values())) < 1e-12
        vs_bad = vs(published, fresh, 10)[0]
        assert vs_bad and not vs(fresh, fresh, 10)[0]
        for k in range(1, 11):
            ratio = k/10
            detected, false_reject, total = 0, 0, 0
            try:
                for subset in itertools.combinations(ids, k):
                    np.random.RandomState = lambda seed, subset=subset: FixedSubset(subset)
                    attack_bad = vp(published, fresh, 10, sample_ratio=ratio, seed=42)[0]
                    honest_bad = vp(fresh, fresh, 10, sample_ratio=ratio, seed=42)[0]
                    assert attack_bad == bool(altered.intersection(subset))
                    detected += int(attack_bad); false_reject += int(honest_bad); total += 1
                    rows.append(dict(round=round_id, changed_weights=m, sampled_clients=k,
                                     sample_ratio=ratio, sample_ids='|'.join(map(str,subset)),
                                     changed_ids='|'.join(map(str,sorted(altered))),
                                     vp_detected=int(attack_bad), vs_detected=int(vs_bad),
                                     vp_honest_rejected=int(honest_bad)))
            finally:
                np.random.RandomState = real_rng
            expected = 1-(math.comb(10-m,k)/math.comb(10,k) if k <= 10-m else 0)
            assert abs(detected/total-expected) < 1e-12 and false_reject == 0
            summaries.append(dict(round=round_id, changed_weights=m, sampled_clients=k,
                                  sample_ratio=ratio, detected=detected, subsets=total,
                                  vp_weight_detection_pct=100*detected/total,
                                  theoretical_pct=100*expected, honest_acceptance_pct=100,
                                  false_rejection_pct=0, vs_weight_detection_pct=100))
            if m == 2:
                for repeat in range(201):
                    start=time.perf_counter_ns()
                    vp(published, fresh, 10, sample_ratio=ratio, seed=round_id*1000+repeat)
                    ns=time.perf_counter_ns()-start
                    timings.append(dict(round=round_id,sampled_clients=k,repeat=repeat,
                                        weight_check_ms=ns/1e6))

for filename, data in [('subsets.csv',rows),('summary.csv',summaries),('timings.csv',timings)]:
    with (OUT/filename).open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
config = dict(purpose=['R4.6: client-sampling-budget sensitivity','R4.13: partial component evidence only'],
              frozen_source=str(SOURCE.relative_to(E)), source_sha256=source_hash,
              captured_training_seed=42,captures=captures,clients=10,
              sampling_counts=list(range(1,11)),changed_weight_counts=[2,4,6],
              transfer_per_pair=0.01,method='Enumerate every size-k subset via a controlled RNG hook in the unchanged frozen function.',
              exclusion='No training, aggregate verification, recovery, distributed execution, or adaptive-attacker claim.',
              np_version=np.__version__,raw_subset_rows=len(rows),summary_rows=len(summaries),timing_rows=len(timings),
              timing_scope='CPU weight comparisons and RNG only; fresh trust inputs already supplied. Not full VP cost.',
              files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*.csv')})
(OUT/'config_and_checks.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
assert len(rows)==9207 and len(summaries)==90 and len(timings)==6030
print(json.dumps(dict(subsets=len(rows),settings=len(summaries),timings=len(timings),
    at_three_clients=[r for r in summaries if r['round']==1 and r['sampled_clients']==3]),ensure_ascii=False))
