from pathlib import Path
import sys,unittest
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset,DataLoader
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT),str(ROOT/'core')]
from scripts.exp3 import exp3_governance_tasv2 as gov
class FinalRoles(unittest.TestCase):
 def run_round(self,group):
  loader=DataLoader(TensorDataset(torch.zeros(4,1,28,28),torch.zeros(4,dtype=torch.long)),batch_size=4)
  original=gov.full_trust;calls=[]
  def counted(*args,**kwargs):calls.append(kwargs.get('return_history',False));return original(*args,**kwargs)
  with patch.object(gov,'full_trust',side_effect=counted):
   r=gov.run_experiment_group(group,[loader]*3,loader,1,{1:gov.RoundAttack(True,'A1')},torch.device('cpu'),42,byzantine_clients=[0],client_attack='label_flip',exclusion_budget=1)
  return r.round_logs[0],calls
 def test_full_tas_uses_vs_reference_for_recovery_without_second_kernel(self):
  row,calls=self.run_round('trust_v1v2')
  self.assertEqual(len(calls),2)
  self.assertEqual(row['healed_applied'],1);self.assertEqual(row['skipped'],0)
  self.assertEqual(row['recovery_v1_endorsed'],1);self.assertEqual(row['recovery_v2_endorsed'],1)
 def test_vp_only_reconstructs_once_on_recovery_without_trusting_executor_weights(self):
  row,calls=self.run_round('trust_v2');self.assertEqual(len(calls),2)
  self.assertEqual(row['healed_applied'],1);self.assertEqual(row['recovery_v2_endorsed'],1)
 def test_cache_binding_covers_input_history_config_and_round(self):
  import copy,numpy as np
  u={0:{'w':torch.ones(2)}};s=({0:1.},[],None,np.ones(2),set(),{})
  key=gov.audit_input_key(1,'parent',u,s,{'alpha':.7})
  for r,p,x,h,c in [(2,'parent',u,s,{'alpha':.7}),(1,'other',u,s,{'alpha':.7}),(1,'parent',{0:{'w':torch.zeros(2)}},s,{'alpha':.7}),(1,'parent',u,({0:.5},[],None,np.ones(2),set(),{}),{'alpha':.7}),(1,'parent',u,s,{'alpha':.5})]:
   self.assertNotEqual(key,gov.audit_input_key(r,p,x,h,c))
if __name__=='__main__':
 torch.set_num_threads(2);unittest.main(verbosity=2)
