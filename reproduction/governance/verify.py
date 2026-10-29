"""Check released governance records without training or tensor payloads."""
from pathlib import Path
import hashlib,json,math,statistics

BASE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    manifest=read(BASE/'FILE_MANIFEST.json')
    for name,entry in manifest.items():
        p=BASE/name;assert p.stat().st_size==entry['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256'],name
    counts=[];observations=[]
    for cap in [0,4]:
        root=BASE/'records'/('cap'+str(cap));code=BASE/'code'/('cap'+str(cap))
        for dataset in ['mnist','fashionmnist']:
            out=root/dataset if cap==0 else root
            run=read(out/'run_manifest.json');cfg=run['configuration'];assert cfg['exclusion_cap']==cap and cfg['exclusion_enabled']==(cap>0)
            for name in ['run_matrix.py','run_experiments.py','core/model.py','scripts/exp2/exp2_run.py','scripts/exp3/exp3_governance_tasv2.py','state_digest.py']:
                assert hashlib.sha256((code/name).read_bytes()).hexdigest()==run['source_sha256'][name],name
            partition=read(out/(dataset+'_partition_indices.json'));indices=[x for ids in partition.values() for x in ids];assert len(indices)==len(set(indices))==60000 and min(indices)==0 and max(indices)==59999
            for attack in ['label_flip','scaling']:
                scenario=out/'matrix'/dataset/attack;ref=read(scenario/'reference/round_metrics.json');learning=read(scenario/'learning_entry_rounds.json')
                assert len(ref)==len(learning)==50 and all(a['state_hash']==b['state_hash'] for a,b in zip(ref,learning))
                for group in ['reference','trust_only','vs_only','vp_only','full_tas']:
                    rr=read(scenario/group/'round_metrics.json');ss=read(scenario/group/'summary.json');assert len(rr)==50
                    avg=statistics.mean(x['accuracy'] for x in rr);assert math.isclose(avg,ss['avg_accuracy'],abs_tol=1e-9)
                    attacked=[x for x in rr if x['attack_active']];honest=[x for x in rr if not x['attack_active']]
                    reject=lambda x:any(x[k] for k in ['v1_trust','v1_agg','v2_trust','v2_agg'])
                    tp=sum(reject(x) for x in attacked);fp=sum(reject(x) for x in honest)
                    assert tp==ss['true_rejections'] and fp==ss['false_rejections']
                    if group!='reference':assert len(attacked)==30 and len(honest)==20
                    for x in rr:
                        assert not x['skipped'] and x['proposal_binding_valid']
                        norms={int(k):v for k,v in json.loads(x['update_norms']).items()};weights={int(k):v for k,v in json.loads(x['final_weights']).items()};med=statistics.median(norms.values())
                        assert abs(sum(weights.values())-1)<1e-10 and sum(w for c,w in weights.items() if norms[c]>2.5*med)==0
                        if cap==0:
                            assert set(norms)==set(range(10)) and not json.loads(x['excluded_clients'])
                            if x['attack_active']:assert {0,1,2,3}.issubset(set(json.loads(x['attack_targets'])))
                        else:assert len(json.loads(x['excluded_clients']))<=4
                    matches=sum(x['state_hash']==y['state_hash'] for x,y in zip(rr,ref))
                    if group=='full_tas':assert tp==30 and fp==0 and matches==50 and all(x['accuracy']==y['accuracy'] for x,y in zip(rr,ref))
                    observations.append(dict(budget=cap,dataset=dataset,attack=attack,configuration=group,avg=avg,intercepted=tp,attacked=len(attacked),honest_accepted=len(honest)-fp,honest=len(honest),complete_state_matches=matches))
                counts.append(dict(budget=cap,dataset=dataset,attack=attack,full_tas_states=50,learning_entry_states=50))
    report=dict(files_verified=len(manifest),conditions=len(observations),scenarios=len(counts),full_tas_complete_state_matches=sum(x['full_tas_states'] for x in counts),learning_entry_complete_state_matches=sum(x['learning_entry_states'] for x in counts),observations=observations)
    print(json.dumps(report,indent=2));return report
if __name__=='__main__':main()
