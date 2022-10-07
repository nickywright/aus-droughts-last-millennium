# aus-droughts-lastmill
Scripts for analysing drought frequency in Australia in last millennium models


## Requirements:
- `numpy`
- `xarray`
- `netCDF4`
- `matplotlib`
- `pandas`
- `cftime`
- `bottleneck`
- `xesmf` for regridding. (doesn't natively support M1 mac right now (https://github.com/pangeo-data/xESMF/issues/165) so you need to use an x86 env)
- `dask`
- `regionmask`


## Scripts:
[analysis/drought_frequencies-processing.py](analysis/drought_frequencies-processing.py): Main script to process files from PMIP3 and CESM-LME(?).
- Subsets all data up to 2000
- Calculates drought metrics using definitions in [climate_droughts_xr_funcs.py](climate_droughts_xr_funcs.py).
  - 2S2E
  - Below a threshold
  - etc
