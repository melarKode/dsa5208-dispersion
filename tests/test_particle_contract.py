"""Exercise legacy particle output compatibility and comparison safeguards."""
import sys
import tempfile
import unittest
from pathlib import Path
import shutil
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from results import field
from compare import compare

SAMPLE=Path(__file__).resolve().parents[1]/'lagrangian/examples/sample_output/gaussian_uniform'

class ParticleContract(unittest.TestCase):
    def test_all_sample_snapshots(self):
        for i,t in enumerate([0,.5,1,1.5,2]):
            cfg,time,f=field(SAMPLE,i)
            self.assertEqual(time,t)
            self.assertEqual(f.shape,(32,32,32))
            self.assertAlmostEqual(float(f.sum()),1.)
            self.assertTrue((f>=0).all())
            self.assertEqual(cfg['solver_family'],'particle')

    def test_native_equivalence_and_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'native';shutil.copytree(SAMPLE,p)
            (p/'effective_config.txt').rename(p/'effective.cfg')
            manifest=p/'snapshots.csv'
            manifest.write_text(manifest.read_text().replace('snapshot_index','index'))
            self.assertEqual(compare(p,SAMPLE)['linf'],0)
            cfg=p/'effective.cfg';cfg.write_text(cfg.read_text().replace('diffusivity=0.1','diffusivity=0.2'))
            with self.assertRaisesRegex(ValueError,'diffusivity'):compare(p,SAMPLE)

    def test_missing_cell_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'bad';shutil.copytree(SAMPLE,p)
            path=p/'concentration_000004.csv'
            path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'cell IDs'):field(p)

if __name__=='__main__':unittest.main()
