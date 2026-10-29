import unittest
from unittest.mock import patch
import numpy as np
import torch
from scripts.exp3 import exp3_governance_tasv2 as gov
from sampled_weight_audit import lightweight_weight_crosscheck


class LightweightCrosscheck(unittest.TestCase):
    def truth(self,updates,state):
        ema,history,anchor,temporal,_,_=state
        return gov.full_trust({c:gov.flatten_update(u) for c,u in updates.items()},
            trust_power=5.,max_weight_ratio=1.5,norm_penalty_strength=.8,refine_anchor=True,
            ema_weights=ema,ema_alpha=.7,cos_history=history,prev_anchor=anchor,temporal_anchor=temporal)[0]

    def test_crosscheck_does_not_recompute_the_full_trust_kernel(self):
        updates={i:{'weight':torch.tensor([1.,.01*i])} for i in range(10)}
        state=({i:.1 for i in updates},[],None,np.array([1.,0.]),set(),{})
        published=self.truth(updates,state)
        with patch.object(gov,'full_trust',side_effect=AssertionError('A lightweight cross-check must not call the full trust kernel')):
            self.assertFalse(lightweight_weight_crosscheck(published,updates,state,10,42)[0])

    def test_honest_outputs_are_accepted_under_heterogeneous_histories(self):
        for seed in range(24):
            rng=np.random.default_rng(seed);count=[3,6,10][seed%3]
            ids=list(range(10-count,10))
            updates={i:{'weight':torch.from_numpy((rng.normal(size=37)+.5).astype(np.float32))} for i in ids}
            if seed%4==0:updates[ids[0]]['weight']*=20.
            ema=None if seed%5==0 else {i:float(v) for i,v in enumerate(rng.dirichlet(np.ones(10)))}
            temporal=rng.normal(size=37);anchor=rng.normal(size=37)
            state=(ema,[],anchor,temporal,set(range(10-count)),{})
            published=self.truth(updates,state)
            self.assertFalse(lightweight_weight_crosscheck(published,updates,state,10,42+seed,sample_ratio=1.)[0],seed)

    def test_norm_inadmissible_positive_weight_is_detected(self):
        updates={0:{'weight':torch.tensor([100.,0.])},1:{'weight':torch.tensor([1.,0.])},2:{'weight':torch.tensor([1.,0.])}}
        state=({0:1/3,1:1/3,2:1/3},[],None,np.array([1.,0.]),set(),{})
        self.assertTrue(lightweight_weight_crosscheck({0:.1,1:.45,2:.45},updates,state,3,42,sample_ratio=1.)[0])

    def test_zero_direction_positive_weight_is_detected_when_fallback_is_impossible(self):
        updates={0:{'weight':torch.tensor([-1.,0.])},1:{'weight':torch.tensor([1.,0.])},2:{'weight':torch.tensor([1.,0.])}}
        state=({0:1/3,1:1/3,2:1/3},[],None,np.array([1.,0.]),set(),{})
        self.assertTrue(lightweight_weight_crosscheck({0:.1,1:.45,2:.45},updates,state,3,42,sample_ratio=1.)[0])

    def test_honest_zero_direction_fallback_is_not_rejected(self):
        updates={i:{'weight':torch.tensor([1.,.01*i])} for i in range(3)}
        state=({i:1/3 for i in updates},[],None,np.array([-1.,0.]),set(),{})
        published=self.truth(updates,state)
        self.assertFalse(lightweight_weight_crosscheck(published,updates,state,3,42,sample_ratio=1.)[0])

    def test_client_sampling_matches_the_existing_rule(self):
        updates={i:{'weight':torch.tensor([1.,0.])} for i in range(10)}
        state=({i:.1 for i in updates},[],None,np.array([1.,0.]),set(),{})
        _,sampled,_=lightweight_weight_crosscheck({i:.1 for i in updates},updates,state,10,4321)
        self.assertEqual(sampled,set(np.random.RandomState(4321).choice(10,size=3,replace=False).tolist()))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
