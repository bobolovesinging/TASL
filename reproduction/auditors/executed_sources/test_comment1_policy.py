"""C1 regressions for the minimal norm mask and norm-constrained zero-direction branch."""
from pathlib import Path
import sys,unittest
import numpy as np
import torch
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp2 import exp2_run as m

class Comment1Policy(unittest.TestCase):
    def test_history_does_not_restore_norm_rejected(self):
        updates={i:np.array([100. if i<4 else 1.,0.]) for i in range(10)}
        w,_,_=m.compute_tasl_final_weights(updates,ema_weights={i:.1 for i in updates})
        self.assertEqual(sum(w[i] for i in range(4)),0.)
        self.assertAlmostEqual(sum(w.values()),1.)

    def test_similarity_smoothing_and_direction_do_not_make_disjoint_hard_masks(self):
        updates={i:np.array([1.,0.]) for i in range(10)}
        updates[0]=np.array([-.5,1.])
        w,_,_=m.compute_tasl_final_weights(updates,ema_weights={i:.1 for i in updates},temporal_anchor=np.array([-1.,0.]))
        self.assertAlmostEqual(sum(w.values()),1.)
        self.assertGreater(w[0],0.)

    def test_zero_direction_fallback_never_restores_norm_rejected(self):
        updates={i:np.array([100. if i<4 else 1.,0.]) for i in range(10)}
        w,_,_=m.compute_tasl_final_weights(updates,ema_weights={i:.1 for i in updates},temporal_anchor=np.array([-1.,0.]))
        self.assertAlmostEqual(sum(w.values()),1.)
        self.assertEqual(sum(w[i] for i in range(4)),0.)

    def test_pre_direction_history_is_preserved(self):
        updates={i:np.array([1.,i*.1]) for i in range(10)}
        kw=dict(ema_weights={i:.1 for i in updates},temporal_anchor=np.array([1.,1.]))
        expected,_,_=m.compute_tasl_trust_weights(updates,**kw)
        w,_,_,history=m.compute_tasl_final_weights(updates,return_history=True,**kw)
        self.assertEqual(expected,history)

    def test_amplification_grid_obeys_norm_bound(self):
        for factor in [1.,3.,10.,100.,1000.,10000.]:
            updates={i:np.array([factor if i<4 else 1.,0.]) for i in range(10)}
            for ref in [np.array([1.,0.]),np.array([-1.,0.])]:
                w,_,_=m.compute_tasl_final_weights(updates,ema_weights={i:.1 for i in updates},temporal_anchor=ref)
                aggregate=sum(w[c]*v for c,v in updates.items())
                self.assertLessEqual(np.linalg.norm(aggregate),2.5*np.median([np.linalg.norm(v) for v in updates.values()])+1e-12)
                if factor>2.5:self.assertEqual(sum(w[c] for c in range(4)),0.)

    def test_nonfinite_round_preserves_full_parent_state(self):
        from scripts.exp3 import exp3_governance_tasv2 as gov
        from torch.utils.data import TensorDataset,DataLoader
        loader=DataLoader(TensorDataset(torch.zeros(2,1,28,28),torch.zeros(2,dtype=torch.long)),batch_size=2)
        def invalid(**kw):
            return {key:value+torch.full_like(value,float('nan')) for key,value in kw['global_state'].items()}
        with patch.object(gov,'train_one_client',side_effect=invalid):
            result=gov.run_experiment_group('noexecutor',[loader]*3,loader,1,{},torch.device('cpu'),42,client_attack='none')
        row=result.round_logs[0]
        self.assertEqual(row['skipped'],1)
        self.assertEqual(row['state_hash'],row['parent_hash'])

    def test_zero_updates_remain_bounded(self):
        updates={i:np.zeros(2) for i in range(10)}
        w,_,_=m.compute_tasl_final_weights(updates)
        self.assertAlmostEqual(sum(w.values()),1.)
        self.assertEqual(float(np.linalg.norm(sum(w[c]*v for c,v in updates.items()))),0.)

if __name__=='__main__':unittest.main(verbosity=2)
