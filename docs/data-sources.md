# Dataset attribution

Alejandro González, María Isabel Castillo Catalán, Sergio Chiva and Sergio Iserte (2026), *Distributed Training of U-Net Architecture for Data-Driven Aerodynamic Flow and Pollutant Dispersion Prediction in Urban Environments — Dataset*.

- Source: https://doi.org/10.5281/zenodo.21674773
- Dataset licence: Creative Commons Attribution 4.0, https://creativecommons.org/licenses/by/4.0/
- Original metadata: `data/metadata/zenodo-21674773.json`.
- `data/geometry/00-buildings.stl` is the original `cell_centers/buildings.stl` archive member. Its SHA256 and source URL are in `results/atlas-geometry-1031990.venus01/geometry-source.json`.
- The candidate mask, cropped case and plots were derived from this geometry and the dataset wind. The report describes the crop, face reconstruction and geometry limitations.
- Full upstream wind archives are downloaded separately with `scripts/download_cluster.sh`.

Kokkos is fetched separately at the pinned upstream revision; its upstream licence accompanies the download. This submission does not redistribute Kokkos source.
