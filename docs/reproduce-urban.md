# Reproduce the recorded urban experiment

Use an Atlas allocation and the Python/compiler environment from the README. Replace recorded upstream job IDs with your new paths as each stage completes.

1. `bash scripts/download_cluster.sh` downloads the wind and verifies its checksum. The dataset is Zenodo record 21674773. The required building geometry `data/geometry/00-buildings.stl` and candidate mask are included in this submission. The original geometry came from the dataset archive. The large upstream archives are deliberately not bundled.
2. Reconstruct the candidate mask with `scripts/build_stl_mask.py` (see its `--help`) or use the included small `data/geometry/building-mask.h5` with its `mask-report.json`. The included candidate is the one used in the report; its geometry limitations are discussed there.
3. Submit `scripts/atlas_stl_audit.pbs`. It replaces the supplied mask in a temporary wind copy and prepares the velocity/mask binaries. Record the `data/prepared/sangeronimo-stl-JOBID` path from the log.
4. In `scripts/atlas_urban_pilot.pbs`, replace `sangeronimo-stl-1031998.venus01` with that path. Submit and retain its new cropped directory (`data/prepared/urban-cropped-JOBID`). The pilot includes the short grid serial/MPI equivalence check.
5. In `scripts/atlas_urban_main.pbs`, replace the historical cropped case reference with the new directory. Submit the 600-second grid experiment. Keep its full outputs on Atlas; copy the compact review directory for reporting.
6. In `scripts/atlas_urban_particles.pbs`, replace the historical grid experiment and cropped case paths with the new paths. Submit the three-seed and sensitivity comparisons. Its serial seed12345 output is the reference for the MPI replay.
7. In `scripts/atlas_urban_particle_mpi.pbs`, replace the historical serial experiment path with the new seed12345 directory. Submit and inspect `review/equivalence.json`, not just job status.

Before submitting each stage, search for historical dependencies:

```bash
rg -n '103[0-9]+|data/prepared|results/atlas' scripts/atlas_urban*.pbs
```

The solvers use the same prepared `case.cfg`, `u.bin`, `v.bin`, `w.bin`, and `solid.bin`. Check input.sha256 before comparing. The recorded source is (150,550,12) m, Gaussian sigma 3 m, initial mass 1, diffusivity 1 m²/s, open x/y and closed z boundaries. The grid selects a safe explicit timestep. The main particles use dt=0.1 s and 100,000 particles; sensitivity runs use dt=0.05 s and 200,000 particles separately. Preserve output times (every 30 s through 600 s).

The supplied mask was rejected as a building map. Do not substitute `atlas_prepare_urban.pbs`'s provisional supplied-mask result for the STL-derived case. Archive input checksums and effective configs for every reproduction; different geometry or wind is a different experiment.
