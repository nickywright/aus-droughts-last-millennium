# aus-droughts-lastmill
Scripts for analysing drought frequency in Australia in last millennium models


Requirements:
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