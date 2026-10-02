"""R4.6 one-factor sensitivity; no change to the frozen training source."""
import argparse,csv,hashlib,importlib.util,inspect,json,textwrap
from pathlib import Path
import numpy as np
import torch,yaml
ROOT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--group',choices=['exposed','anchor_norm'],required=True);a=ap.parse_args()
source=ROOT/'pipeline/scripts/exp2/exp2_run.py'
assert hashlib.sha256(source.read_bytes()).hexdigest()=='8f4c447acab1f78e750592825643954dcf9ed8ba7402c907ca2cafe272a2ef83'
spec=importlib.util.spec_from_file_location('frozen_sweep',source);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
cfg=yaml.safe_load((ROOT/'seed42.yaml').read_text());p.set_seed(42);torch.set_num_threads(4)
loaders,test=p.load_fashionmnist_data(cfg);model_fn=lambda:p.create_model(cfg)
out=ROOT/'trust_sweep'/a.group;out.mkdir(parents=True,exist_ok=False)
original=p.compute_tasl_trust_weights

def variant(blend=0.6,gate=2.5):
    s=textwrap.dedent(inspect.getsource(original))
    for old,new in [
      ('norm_gate = median_norm * 2.5',f'norm_gate = median_norm * {gate!r}'),
      ('anchor = 0.6 * anchor + 0.4 * consensus_anchor',f'anchor = {blend!r} * anchor + {1-blend!r} * consensus_anchor'),
      ('anchor = 0.6 * refined_anchor + 0.4 * prev_anchor',f'anchor = {blend!r} * refined_anchor + {1-blend!r} * prev_anchor'),
      ('anchor = 0.6 * anchor + 0.4 * temporal_anchor',f'anchor = {blend!r} * anchor + {1-blend!r} * temporal_anchor')]:
        assert s.count(old)==1; s=s.replace(old,new)
    ns=dict(p.__dict__);exec(compile(s,'<sensitivity-only-trust-function>','exec'),ns)
    return ns['compute_tasl_trust_weights'],s

# Default transformation must preserve the deployed result, including history.
# Spell the original complementary coefficient exactly, preventing a test from
# concealing a changed baseline due to 1-0.6 floating-point representation.
same,same_src=variant();rng=np.random.RandomState(27)
u={i:rng.randn(32) for i in range(10)};hist={i:0.1 for i in range(10)}
kwargs={'ema_weights':hist,'prev_anchor':rng.randn(32),'temporal_anchor':rng.randn(32)}
ref=original(u,**kwargs);actual=same(u,**kwargs)
assert ref[0]==actual[0] and ref[1]==actual[1] and np.array_equal(ref[2],actual[2])
cases=([('trust_power',3.0),('trust_power',7.0),('norm_penalty',0.4),('norm_penalty',1.2),
        ('weight_cap_ratio',1.0),('weight_cap_ratio',2.0)] if a.group=='exposed' else
       [('anchor_current_fraction',0.4),('anchor_current_fraction',0.8),('norm_gate_multiplier',2.0),('norm_gate_multiplier',3.0)])
rows=[]
for parameter,value in cases:
 c=yaml.safe_load((ROOT/'seed42.yaml').read_text())
 if parameter in ('anchor_current_fraction','norm_gate_multiplier'):
    p.compute_tasl_trust_weights,code=variant(blend=value if parameter=='anchor_current_fraction' else 0.6,
                                            gate=value if parameter=='norm_gate_multiplier' else 2.5)
    (out/f'{parameter}_{value}.py').write_text(code)
 else:c['algorithm_params']['tasl'][parameter]=value;p.compute_tasl_trust_weights=original
 result=p.run_single('tasl_full','label_flip',loaders,test,c,model_fn,10,4,50,torch.device('cuda'))
 (out/f'{parameter}_{value}.json').write_text(json.dumps({'config':c,'parameter':parameter,'value':value,'result':result},indent=2))
 rows.append({'parameter':parameter,'value':value,'avg_acc':result['avg_acc'],'avg_asr':result['avg_asr'],
              'honest_zero_weights':sum(r['honest_rejected'] for r in result['round_metrics']),
              'honest_client_rounds':300})
 with (out/'summary.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('COMPLETED',json.dumps(rows[-1]),flush=True)
(out/'complete.json').write_text(json.dumps({'comment':'R4.6','rounds':50,'seed':42,'cases':len(cases),
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'default_transform_equivalence':True,
 'only_one_factor_varied':True,'clean_training_not_scanned':True,'no_test_set_selection':True},indent=2))
