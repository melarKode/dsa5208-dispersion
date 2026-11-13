"""Check conservative binning, edge handling and normalized comparison metrics."""
import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from aggregate_urban_comparison import block_sum,metrics

class AggregationTests(unittest.TestCase):
 def test_partial_edges(self):
  a=np.arange(3*5*7,dtype=float).reshape(3,5,7)
  b=block_sum(a,4)
  self.assertEqual(b.shape,(1,2,2));self.assertEqual(b.sum(),a.sum())
  self.assertEqual(b[0,1,1],a[:,4:,4:].sum())
 def test_identity(self):
  a=np.arange(24.).reshape(2,3,4);np.testing.assert_array_equal(block_sum(a,1),a)
 def test_nested_contraction(self):
  rng=np.random.default_rng(7);a=rng.random((5,13,17));b=rng.random(a.shape)
  errors=[metrics(block_sum(a,f),block_sum(b,f))['l1_mass'] for f in [1,2,4,8,16,32]]
  self.assertTrue(np.all(np.diff(errors)<=1e-12))
 def test_shape_separates_mass(self):
  m=metrics(np.array([1.,3.]),np.array([2.,6.]))
  self.assertAlmostEqual(m['normalized_l1'],1/3);self.assertEqual(m['survivor_total_variation'],0)
 def test_disjoint(self):
  m=metrics(np.array([1.,0.]),np.array([0.,1.]))
  self.assertEqual(m['normalized_l1'],1);self.assertEqual(m['survivor_total_variation'],1)
 def test_zero(self):
  self.assertIsNone(metrics(np.zeros(2),np.ones(2))['survivor_total_variation'])
 def test_reject_invalid(self):
  with self.assertRaises(ValueError):block_sum(np.ones(3),0)
  with self.assertRaises(ValueError):metrics(np.array([-1.]),np.array([1.]))

if __name__=='__main__':unittest.main()
