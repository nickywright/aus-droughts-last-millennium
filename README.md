# aus-droughts-last-millennium

Scripts for analysing multi-year drought metrics in Australia in last millennium models. 


Questions?? Contact <nicky.wright@sydney.edu.au> and/or <georgina.falster@anu.edu.au>

## Citation
Falster, G. M., Wright, N. M., Abram, N. J., Ukkola, A. M., and Henley, B. J., 2024, Potential for historically unprecedented Australian droughts from natural variability and climate change, *Hydrology and Earth System Sciences*, 28, 1383–1401, https://doi.org/10.5194/hess-28-1383-2024.


## Files contained in this repository:
- `climate_xr_funcs.py`: some helpful functions for reading in climate models
- `climate_droughts_xr_funcs.py`: some functions specifically related to calculating the drought metrics
- `analysis/drought_frequencies-processing.py`: main script to process all files from PMIP3 and CESM-LME models:
	- Subsets all data up to 2000
	- Calculates precipitation anomalies, relative to specified climatology period (i.e. 1900-2000 mean)
	- Calculates drought metrics (length, intensity, severity, frequency) using definitions in [climate_droughts_xr_funcs.py](climate_droughts_xr_funcs.py).
	  - '2S2E': a drought starts after two years of negative precipitation anomalies, and ends after two years of positive precipitation anomalies (used within the paper)
	  - below a threshold in general (e.g. 20%, median) of climatology
	- Saves file per model as netcdf
- Scripts for figures 1 to 10
- Murray-Darling Basin (MDB) shapefile


