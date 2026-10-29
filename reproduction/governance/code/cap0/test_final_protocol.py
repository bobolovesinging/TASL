from pathlib import Path
import sys,unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset,DataLoader
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp3 import exp3_governance_tasv2 as gov

class RoleRecomputation(unittest.TestCase):
 def test_two_role_references_and_recovery_are_independent_calls(self):
  loader=DataLoader(TensorDataset(torch.zeros(4,1,28,28),torch.zeros(4,dtype=torch.long)),batch_size=4)
  original=gov.full_trust;calls=[]
  def counted(*args,**kwargs):calls.append(kwargs.get('return_history',False));return original(*args,**kwargs)
  with patch.object(gov,'full_trust',side_effect=counted):
   r=gov.run_experiment_group('trust_v1v2',[loader]*3,loader,1,{1:gov.RoundAttack(True,'A1')},torch.device('cpu'),42,
      byzantine_clients=[0],client_attack='label_flip',exclusion_budget=1)
  self.assertEqual(len(calls),6)
  self.assertEqual(sum(calls),1)
  self.assertEqual(r.round_logs[0]['healed_applied'],1)
  self.assertEqual(r.round_logs[0]['skipped'],0)

if __name__=='__main__':unittest.main(verbosity=2)
