# Spatially aggregated comparison

This postprocessing reads existing grid job 1032157 and particle job 1032182 snapshots at 120, 300 and 600 seconds. It does not run either solver.

Submit `scripts/atlas_urban_aggregate.pbs` after copying it and `scripts/aggregate_urban_comparison.py` to Atlas. Existing `scripts/results.py` and `scripts/compare.py` are dependencies. Output is `results/atlas-aggregate-JOBID` (compact maps, CSV metrics, JSON provenance/checks).

To plot the saved output:

```bash
MPLCONFIGDIR=/tmp/dispersion-mpl .venv/bin/python scripts/plot_urban_aggregate.py results/atlas-aggregate-JOBID
```

Default map bins are 8 by 8 horizontal cells, approximately 18.7 by 18.7 metres. Use `--factor 4`, `16` or `32` for other saved scales. Map values are column mass divided by each bin's exact footprint area. Partial edge bins are retained. Separate 3D metrics use cubic index blocks, with physical widths set by unequal cell spacings; bins can span the full vertical domain at the largest factor.

Metrics include raw mass L1, L1 normalized by total remaining mass across both solvers, and total variation between separately normalized survivor distributions. The last metric separates spatial shape from differences in total remaining mass. Undefined survivor shapes (zero remaining mass) are recorded as null, not zero error.

All three particle seeds are measured individually and averaged as fields before measuring ensemble error. Half-timestep and double-particle-count runs remain individual-seed sensitivity cases. Three seeds are insufficient for strong uncertainty claims.

Coarsening necessarily reduces or preserves L1 on nested bins. Improvement is therefore expected mathematically and must not be described as proof of physical accuracy or solver convergence. Column integration also hides vertical differences; report 3D metrics alongside maps.

The bundled Atlas job 1036035 contains 180 metric rows. `scripts/audit_report_evidence.py` checks native comparisons and exported coarse-map masses and discrepancies.
