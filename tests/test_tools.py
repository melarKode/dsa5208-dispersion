"""Check inspection, conversion, audit, and comparison failure modes."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_case import audit
from inspect_dataset import inspect
from prepare_case import prepare
from compare import compare


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, static_mask=True):
        path = self.root / "input.h5"
        with h5py.File(path, "w") as f:
            f["wind"] = np.ones((1, 4, 3, 2, 3))
            f["solid"] = np.zeros((4, 3, 2) if static_mask else (1, 4, 3, 2), dtype="u1")
            f.attrs["units"] = "m/s"
        return path

    def arguments(self, path):
        return argparse.Namespace(file=path, output=self.root / "case", axis_order="xyz", dx=1., dy=2., dz=3.,
                                  origin=[0., 0., 0.], case_index=0, velocity="wind", component_axis=-1,
                                  components=[0, 1, 2], u=None, v=None, w=None, solid="solid",
                                  solid_value=1, air_value=0, all_fluid=False)

    def test_static_mask_and_divergence(self):
        path = self.fixture()
        report = inspect(path, True)
        self.assertEqual(report["objects"]["/wind"]["statistics"]["count"], 72)
        prepare(self.arguments(path))
        report = audit(self.root / "case", ["periodic"] * 3)
        self.assertEqual(report["max_abs_divergence"], 0)
        self.assertEqual(report["air_cells"], 24)
        self.assertTrue((self.root / "case/input.sha256").exists())
        self.assertFalse((self.root / "case/INCOMPLETE").exists())

    def test_invalid_air_velocity_leaves_marker(self):
        path = self.fixture()
        with h5py.File(path, "r+") as f:
            f["wind"][0, 0, 0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            prepare(self.arguments(path))
        self.assertTrue((self.root / "case/INCOMPLETE").exists())
        with self.assertRaisesRegex(ValueError, "incomplete"):
            audit(self.root / "case", ["periodic"] * 3)

    def test_nonbinary_mask(self):
        path = self.fixture()
        with h5py.File(path, "r+") as f:
            f["solid"][0, 0, 0] = 2
        with self.assertRaisesRegex(ValueError, "Mask contains"):
            prepare(self.arguments(path))

    def test_truncated_hdf5(self):
        path = self.root / "bad.h5"
        path.write_bytes(b"not a valid HDF5 file")
        with self.assertRaises(OSError):
            inspect(path)

    def test_particle_csv_contract(self):
        for side in ("a", "b"):
            folder = self.root / side
            folder.mkdir()
            (folder / "effective.cfg").write_text("nx=2\nny=1\nnz=1\ndx=1\ndy=1\ndz=1\n")
            (folder / "snapshots.csv").write_text("index,time\n0,0\n")
            (folder / "concentration_000000.csv").write_text("cell_id,i,j,k,mass,concentration\n1,1,0,0,0.4,0.4\n0,0,0,0,0.6,0.6\n")
        result = compare(self.root / "a", self.root / "b")
        self.assertEqual(result["linf"], 0)
        path = self.root / "b/concentration_000000.csv"
        path.write_text(path.read_text().replace("1,1,0,0", "0,1,0,0"))
        with self.assertRaisesRegex(ValueError, "cell IDs"):
            compare(self.root / "a", self.root / "b")


if __name__ == "__main__":
    unittest.main()
