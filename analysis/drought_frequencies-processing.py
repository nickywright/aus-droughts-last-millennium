# %% Code to look at droughts in CESM-LME and PMIP3
# This notebook processes files for looking at droughts


# %% import modules
import sys
sys.path.append('../')  # import functions to make life easier
import climate_xr_funcs
import climate_droughts_xr_funcs

# import things
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cftime
import regionmask
import os
# import salem
import xesmf as xe
from dask.diagnostics import ProgressBar

from scipy import stats


# %% set file path to model output
filepath = '/Volumes/LaCie/CMIP5-PMIP3/CESM-LME/mon/PRECT_v6/'
filepath_pmip3 = '/Volumes/LaCie/CMIP5-PMIP3'

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# some options for running


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Definitions

# ------- get_drought_def

# Import CESM-LME files, subset to 1900 onwards, and calculate droughts
def import_cesmlme(filepath, casenumber):
    casenumber_short = casenumber.lstrip('0')
    ff_precip = climate_xr_funcs.import_full_forcing_precip(filepath, casenumber)
    ff_precip_annual = ff_precip.groupby('time.year').sum('time', skipna=False)
    ff_precip_annual.load()
    return ff_precip_annual


# ------- PMIP3 defs
# Import PMIP3 files
def read_in_pmip3(modelname, var):
    if modelname == 'bcc':
        ds = xr.open_mfdataset('%s/past1000/%s/%s_Amon_bcc-csm1-1_past1000_r1i1p1_*.nc' % (filepath_pmip3, var, var),
                               combine='by_coords', chunks={'time': 1000})
    if modelname == 'ccsm4':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var),
                                chunks={'time': 1000})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_CCSM4_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var),
                                chunks={'time': 1000})
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
        ds_p2['lat'] = ds_p1.lat
        ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'csiro_mk3l':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_CSIRO-Mk3L-1-2_past1000_r1i1p1_085101-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 1000})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_CSIRO-Mk3L-1-2_historical_r1i1p1_185101-200012.nc' % (filepath_pmip3, var, var), chunks={'time': 1000})
        ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'fgoals_gl':
        ds = xr.open_dataset('%s/past1000/%s/%s_Amon_FGOALS-gl_past1000_r1i1p1_100001-199912.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
    if modelname == 'fgoals_s2':
        if var =='ts':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_FGOALS-s2_past1000_r1i1p1_*.nc'  % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 1000})
            ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_FGOALS-s2_historical_r1i1p1_185001-200512.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
            ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        elif var =='pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_FGOALS-s2_past1000_r1i1p1_*.nc'  % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 1000})
            ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_FGOALS-s2_historical_r1i1p1_185001-200512.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
            ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_FGOALS-s2_past1000_r1i1p1_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
            ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_FGOALS-s2_historical_r1i1p1_185001-200512.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
            ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_21':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p121_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p121_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p121_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p121_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_22':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p122_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p122_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p122_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p122_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_23':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p123_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p123_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p123_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p123_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_24':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p124_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p124_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p124_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p124_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_25':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p125_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p125_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p125_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p125_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_26':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p126_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p126_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p126_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p126_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_27':
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p127_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p127_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p127_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p127_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_28':
        if var == 'ts' or var == 'pr' or var == 'psl':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p128_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p128_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p128_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p128_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'hadcm3':
        print('Friendly reminder that hadcm3 is missing years 1801-1859...')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_HadCM3_past1000_r1i1p1_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_HadCM3_historical_r1i1p1_*.nc'  % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
        ds = xr.concat([ds_p1, ds_p2], dim='time')
        # something weird is happening with the time field, re-read it to be consistent just in case
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        ds.update({'time':('time', dates['time'], attrs)})

    if modelname == 'ipsl':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_IPSL-CM5A-LR_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_IPSL-CM5A-LR_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # something weird is happening with the time field, re-read it to be consistent just in case
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        ds.update({'time':('time', dates['time'], attrs)})

    if modelname == 'miroc':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MIROC-ESM_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MIROC-ESM_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        ds.update({'time':('time', dates['time'], attrs)})

    if modelname == 'mpi':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MPI-ESM-P_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MPI-ESM-P_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        ds.update({'time':('time', dates['time'], attrs)})

    if modelname == 'mri':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MRI-CGCM3_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MRI-CGCM3_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, decode_times=False)
        # try and fix the times...
        newdates = cftime.num2date(ds_p2.time.values, 'days since 1850-01-01',  calendar='standard')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}# does this make sense?
        dates = xr.Dataset({'time': ('time', newdates, attrs)})
        # dates = xr.decode_cf(dates)
        ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeGregorian(1851,1,1), drop=True)
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times again
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        ds.update({'time':('time', dates['time'], attrs)})
    return ds

# apply drought metrics
def droughts_historical_fromyear(ds, historical_year, output_dir, model_filename):
    print('... Calculating drought metrics for %s from %s onwards' % (model_filename, historical_year))
    ds_hist = ds.where(ds['year'] >= historical_year, drop=True)    # subset to historical period
    ds_hist_clim = ds_hist.where(ds_hist['year'] <= 2000, drop=True)  # for historical, use up to 2000 for climatology
    # get years that are drought vs not drought
    ds_hist['drought_years_2s2e']       = climate_droughts_xr_funcs.get_drought_years_2S2E_apply(ds_hist.PRECT_mm, ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['drought_years_median']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds_hist.PRECT_mm, ds_hist_clim.PRECT_mm.quantile(0.5, dim=('year')))
    ds_hist['drought_years_20perc']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds_hist.PRECT_mm, ds_hist_clim.PRECT_mm.quantile(0.2, dim=('year')))
    ds_hist['drought_years_120pc_2med'] = climate_droughts_xr_funcs.get_drought_years_120perc_2median_apply(ds_hist.PRECT_mm, ds_hist_clim.PRECT_mm.quantile(0.2, dim='year'), ds_hist_clim.PRECT_mm.quantile(0.5, dim='year'))
    ds_hist['drought_years_220pc_1med'] = climate_droughts_xr_funcs.get_drought_years_start_end_thresholds_apply(ds_hist.PRECT_mm, ds_hist_clim.PRECT_mm.quantile(0.2, dim=('year')), ds_hist_clim.PRECT_mm.quantile(0.5, dim=('year')))
    
    # get overall length of droughts
    ds_hist['droughts_2s2e']       = climate_droughts_xr_funcs.cumulative_drought_length(ds_hist['drought_years_2s2e'])
    ds_hist['droughts_median']     = climate_droughts_xr_funcs.cumulative_drought_length(ds_hist['drought_years_median'])
    ds_hist['droughts_20perc']     = climate_droughts_xr_funcs.cumulative_drought_length(ds_hist['drought_years_20perc'])
    ds_hist['droughts_120pc_2med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds_hist['drought_years_120pc_2med'])
    ds_hist['droughts_220pc_1med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds_hist['drought_years_220pc_1med'])
    
    # get max length in this period
    ds_hist['droughts_2s2e_max']       = climate_droughts_xr_funcs.max_length_ufunc(ds_hist.droughts_2s2e, dim='year')
    ds_hist['droughts_median_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds_hist.droughts_median, dim='year')
    ds_hist['droughts_20perc_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds_hist.droughts_20perc, dim='year')
    ds_hist['droughts_120pc_2med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds_hist.droughts_120pc_2med, dim='year')
    ds_hist['droughts_220pc_1med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds_hist.droughts_220pc_1med, dim='year')
    
    # get mean length in this period
    ds_hist['droughts_2s2e_mean']       = climate_droughts_xr_funcs.mean_length_ufunc(ds_hist.droughts_2s2e, dim='year')
    ds_hist['droughts_median_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds_hist.droughts_median, dim='year')
    ds_hist['droughts_20perc_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds_hist.droughts_20perc, dim='year')
    ds_hist['droughts_120pc_2med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds_hist.droughts_120pc_2med, dim='year')
    ds_hist['droughts_220pc_1med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds_hist.droughts_220pc_1med, dim='year')
    
    # count how many individual events occur
    ds_hist['droughts_2s2e_no_of_events']       = climate_droughts_xr_funcs.count_drought_events_apply(ds_hist.droughts_2s2e)
    ds_hist['droughts_median_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds_hist.droughts_median)
    ds_hist['droughts_20perc_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds_hist.droughts_20perc)
    ds_hist['droughts_120pc_2med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds_hist.droughts_120pc_2med)
    ds_hist['droughts_220pc_1med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds_hist.droughts_220pc_1med)
    
    # std
    ds_hist['droughts_2s2e_std']       = climate_droughts_xr_funcs.std_apply(ds_hist.droughts_2s2e, dim='year')
    ds_hist['droughts_median_std']     = climate_droughts_xr_funcs.std_apply(ds_hist.droughts_median, dim='year')
    ds_hist['droughts_20perc_std']     = climate_droughts_xr_funcs.std_apply(ds_hist.droughts_20perc, dim='year')
    ds_hist['droughts_120pc_2med_std'] = climate_droughts_xr_funcs.std_apply(ds_hist.droughts_120pc_2med, dim='year')
    ds_hist['droughts_220pc_1med_std'] = climate_droughts_xr_funcs.std_apply(ds_hist.droughts_220pc_1med, dim='year')
    
    # intensity  - relative to climatological mean (same as in anna's paper)
    ds_hist['droughts_2s2e_intensity']        = climate_droughts_xr_funcs.drought_intensity(ds_hist, 'drought_years_2s2e', 'droughts_2s2e', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_median_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds_hist, 'drought_years_median', 'droughts_median', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_20perc_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds_hist, 'drought_years_20perc', 'droughts_20perc', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_120pc_2med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds_hist, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_220pc_1med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds_hist, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds_hist_clim.PRECT_mm.mean(dim='year'))
    
    # severity - intensity x length
    ds_hist['droughts_2s2e_severity']        = climate_droughts_xr_funcs.drought_severity(ds_hist, 'drought_years_2s2e', 'droughts_2s2e', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_median_severity']      = climate_droughts_xr_funcs.drought_severity(ds_hist, 'drought_years_median', 'droughts_median', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_20perc_severity']      = climate_droughts_xr_funcs.drought_severity(ds_hist, 'drought_years_20perc', 'droughts_20perc', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_120pc_2med_severity'] = climate_droughts_xr_funcs.drought_severity(ds_hist, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds_hist_clim.PRECT_mm.mean(dim='year'))
    ds_hist['droughts_220pc_1med_severity'] = climate_droughts_xr_funcs.drought_severity(ds_hist, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds_hist_clim.PRECT_mm.mean(dim='year'))
    
    # get rid of quantile
    ds_hist = ds_hist.drop('quantile')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_hist.data_vars}
    
    print('... Saving %s historical file to:  %s' % (model_filename, output_dir))
    
    delayed_obj = ds_hist.to_netcdf('%s/%s_precip_hist_annual.nc' % (output_dir, model_filename), compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()
    
    return ds_hist

def droughts_lm_thresholdyears(ds, threshold_startyear, threshold_endyear, output_dir, model_filename):
    print('... Calculating drought metrics for %s using %s-%s as climatology' % (model_filename, threshold_startyear, threshold_endyear))
    
    ds_lm = ds
    ds_clim = ds_lm.where((ds_lm['year'] >= threshold_startyear) & (ds_lm['year'] <= threshold_endyear), drop=True)
    
    # get years that are drought vs not drought
    ds_lm['drought_years_2s2e']       = climate_droughts_xr_funcs.get_drought_years_2S2E_apply(ds_lm.PRECT_mm, ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['drought_years_median']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds_lm.PRECT_mm, ds_clim.PRECT_mm.quantile(0.5, dim=('year')))
    ds_lm['drought_years_20perc']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds_lm.PRECT_mm, ds_clim.PRECT_mm.quantile(0.2, dim=('year')))
    ds_lm['drought_years_120pc_2med'] = climate_droughts_xr_funcs.get_drought_years_120perc_2median_apply(ds_lm.PRECT_mm, ds_clim.PRECT_mm.quantile(0.2, dim='year'), ds_clim.PRECT_mm.quantile(0.5, dim='year'))
    ds_lm['drought_years_220pc_1med'] = climate_droughts_xr_funcs.get_drought_years_start_end_thresholds_apply(ds_lm.PRECT_mm, ds_clim.PRECT_mm.quantile(0.2, dim=('year')), ds_clim.PRECT_mm.quantile(0.5, dim=('year')))
    
    # get overall length of droughts
    ds_lm['droughts_2s2e']       = climate_droughts_xr_funcs.cumulative_drought_length(ds_lm['drought_years_2s2e'])
    ds_lm['droughts_median']     = climate_droughts_xr_funcs.cumulative_drought_length(ds_lm['drought_years_median'])
    ds_lm['droughts_20perc']     = climate_droughts_xr_funcs.cumulative_drought_length(ds_lm['drought_years_20perc'])
    ds_lm['droughts_120pc_2med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds_lm['drought_years_120pc_2med'])
    ds_lm['droughts_220pc_1med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds_lm['drought_years_220pc_1med'])
    
    # get max length in this period
    ds_lm['droughts_2s2e_max']       = climate_droughts_xr_funcs.max_length_ufunc(ds_lm.droughts_2s2e, dim='year')
    ds_lm['droughts_median_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds_lm.droughts_median, dim='year')
    ds_lm['droughts_20perc_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds_lm.droughts_20perc, dim='year')
    ds_lm['droughts_120pc_2med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds_lm.droughts_120pc_2med, dim='year')
    ds_lm['droughts_220pc_1med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds_lm.droughts_220pc_1med, dim='year')
    
    # get mean length in this period
    ds_lm['droughts_2s2e_mean']       = climate_droughts_xr_funcs.mean_length_ufunc(ds_lm.droughts_2s2e, dim='year')
    ds_lm['droughts_median_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds_lm.droughts_median, dim='year')
    ds_lm['droughts_20perc_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds_lm.droughts_20perc, dim='year')
    ds_lm['droughts_120pc_2med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds_lm.droughts_120pc_2med, dim='year')
    ds_lm['droughts_220pc_1med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds_lm.droughts_220pc_1med, dim='year')
    
    # count how many individual events occur
    ds_lm['droughts_2s2e_no_of_events']       = climate_droughts_xr_funcs.count_drought_events_apply(ds_lm.droughts_2s2e)
    ds_lm['droughts_median_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds_lm.droughts_median)
    ds_lm['droughts_20perc_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds_lm.droughts_20perc)
    ds_lm['droughts_120pc_2med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds_lm.droughts_120pc_2med)
    ds_lm['droughts_220pc_1med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds_lm.droughts_220pc_1med)
    
    # std
    ds_lm['droughts_2s2e_std']       = climate_droughts_xr_funcs.std_apply(ds_lm.droughts_2s2e, dim='year')
    ds_lm['droughts_median_std']     = climate_droughts_xr_funcs.std_apply(ds_lm.droughts_median, dim='year')
    ds_lm['droughts_20perc_std']     = climate_droughts_xr_funcs.std_apply(ds_lm.droughts_20perc, dim='year')
    ds_lm['droughts_120pc_2med_std'] = climate_droughts_xr_funcs.std_apply(ds_lm.droughts_120pc_2med, dim='year')
    ds_lm['droughts_220pc_1med_std'] = climate_droughts_xr_funcs.std_apply(ds_lm.droughts_220pc_1med, dim='year')
    
    # intensity  - relative to climatological mean (same as in anna's paper)
    ds_lm['droughts_2s2e_intensity']        = climate_droughts_xr_funcs.drought_intensity(ds_lm, 'drought_years_2s2e', 'droughts_2s2e', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_median_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds_lm, 'drought_years_median', 'droughts_median', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_20perc_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds_lm, 'drought_years_20perc', 'droughts_20perc', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_120pc_2med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds_lm, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_220pc_1med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds_lm, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds_clim.PRECT_mm.mean(dim='year'))
    
    # severity - intensity x length
    ds_lm['droughts_2s2e_severity']        = climate_droughts_xr_funcs.drought_severity(ds_lm, 'drought_years_2s2e', 'droughts_2s2e', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_median_severity']      = climate_droughts_xr_funcs.drought_severity(ds_lm, 'drought_years_median', 'droughts_median', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_20perc_severity']      = climate_droughts_xr_funcs.drought_severity(ds_lm, 'drought_years_20perc', 'droughts_20perc', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_120pc_2med_severity'] = climate_droughts_xr_funcs.drought_severity(ds_lm, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds_clim.PRECT_mm.mean(dim='year'))
    ds_lm['droughts_220pc_1med_severity'] = climate_droughts_xr_funcs.drought_severity(ds_lm, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds_clim.PRECT_mm.mean(dim='year'))
    
    # get rid of quantile
    ds_lm = ds_lm.drop('quantile')
    
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    print('... Saving %s last millennium file to:  %s' % (model_filename, output_dir))
    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_lm.data_vars}
    
    delayed_obj = ds_lm.to_netcdf('%s/%s_precip_lm_annual.nc' % (output_dir, model_filename), compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()
    
    return ds_lm


def process_pmip3_files(ds, modelname, historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir):
    # import monthly pmip3 and run drought workflow
    # convert monthly to annual
    month_length = xr.DataArray(climate_xr_funcs.get_dpm(ds.pr, calendar='noleap'), coords=[ds.pr.time], name='month_length')
    ds['PRECT_mm'] = ds.pr * 60 * 60 * 24 * month_length # to be in mm/month first
    ds_annual = ds.groupby('time.year').sum('time', skipna=False)
    ds_annual.load()
    
    # process for historical
    ds_hist = droughts_historical_fromyear(ds_annual, historical_year, hist_output_dir, modelname)
    
    # process for lm
    ds_lm = droughts_lm_thresholdyears(ds_annual, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir, modelname)
    
    return ds_hist, ds_lm

def process_cesm_files(ds, modelname, historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir):
    # process for historical
    ds_hist = droughts_historical_fromyear(ds, historical_year, hist_output_dir, modelname)

    # process for lm
    ds_lm = droughts_lm_thresholdyears(ds, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir, modelname)
    return ds_hist, ds_lm	

# Subset to Australia: using regionmask
def get_aus(ds):
    mask = regionmask.defined_regions.natural_earth.countries_110.mask(ds)
    ds_aus = ds.where(mask == 137, drop=True)
    return ds_aus


def save_netcdf_compression(ds, output_dir, filename):

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds.data_vars}

    delayed_obj = ds.to_netcdf('%s/%s.nc' % (output_dir, filename), mode='w', compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------ regrid AWAP to other grid resolution
# this is in preparation of doing some stats tests

# --- DEF
# Instructions for regridding using xesmf are here: https://xesmf.readthedocs.io/en/latest/notebooks/Dataset.html
def regrid_files(ds_to_regrid, ds_target ):
    # resolution of output: same as cesm-lme
    ds_out = xr.Dataset({'lat': (['lat'], ds_target.lat),
                         'lon': (['lon'], ds_target.lon)})

    regridder = xe.Regridder(ds_to_regrid, ds_out, 'bilinear')
    regridder.clean_weight_file()

    ds_out = regridder(ds_to_regrid)
    for k in ds_to_regrid.data_vars:
        print(k, ds_out[k].equals(regridder(ds_to_regrid[k])))

    return ds_out

# Some definitions
# --- DEFs - ks-test
def kstest_2samp(x,y):
    # drop nans - otherwise seems to influence things?
    xx = x[~np.isnan(x)]
    yy = y[~np.isnan(y)]

    # check if array is 0 len
    if (len(xx) == 0) and (len(yy) == 0):
        # both were full of nans, set output to nan
        p_value = np.nan
    elif (len(xx) == 0) or (len(yy) == 0):
        # one is a nan, try the test. Most likely will fail?
        p_value = np.nan
    else:
        tau, p_value = stats.ks_2samp(xx, yy)
    return p_value

def kstest_apply(x,y,dim='year'):
    # dim name is the name of the dimension we're applying it over
    return xr.apply_ufunc(kstest_2samp, x , y, input_core_dims=[[dim], [dim]], vectorize=True,
        output_dtypes=[float],  join='outer' # join='outer' seems to allow unequal lengths for dims-of-interest
        )

def get_significance_droughts_kstest(ds_model, awap):
    ds_sig = xr.Dataset()  # empy ds
    ds_sig['droughts_2s2e'] = kstest_apply(ds_model.droughts_2s2e, awap.droughts_2s2e)
    ds_sig['droughts_2s2e_intensity'] = kstest_apply(ds_model.droughts_2s2e_intensity, awap.droughts_2s2e_intensity)
    ds_sig['droughts_2s2e_severity'] = kstest_apply(ds_model.droughts_2s2e_severity, awap.droughts_2s2e_severity)

    ds_sig['droughts_median'] = kstest_apply(ds_model.droughts_median, awap.droughts_median)
    ds_sig['droughts_median_intensity'] = kstest_apply(ds_model.droughts_median_intensity, awap.droughts_median_intensity)
    ds_sig['droughts_median_severity'] = kstest_apply(ds_model.droughts_median_severity, awap.droughts_median_severity)

    ds_sig['droughts_20perc'] = kstest_apply(ds_model.droughts_20perc, awap.droughts_20perc)
    ds_sig['droughts_20perc_intensity'] = kstest_apply(ds_model.droughts_20perc_intensity, awap.droughts_20perc_intensity)
    ds_sig['droughts_20perc_severity'] = kstest_apply(ds_model.droughts_20perc_severity, awap.droughts_20perc_severity)

    ds_sig['droughts_120pc_2med'] = kstest_apply(ds_model.droughts_120pc_2med, awap.droughts_120pc_2med)
    ds_sig['droughts_120pc_2med_intensity'] = kstest_apply(ds_model.droughts_120pc_2med_intensity, awap.droughts_120pc_2med_intensity)
    ds_sig['droughts_120pc_2med_severity'] = kstest_apply(ds_model.droughts_120pc_2med_severity, awap.droughts_120pc_2med_severity)

    ds_sig['droughts_220pc_1med'] = kstest_apply(ds_model.droughts_220pc_1med, awap.droughts_220pc_1med)
    ds_sig['droughts_220pc_1med_intensity'] = kstest_apply(ds_model.droughts_220pc_1med_intensity, awap.droughts_220pc_1med_intensity)
    ds_sig['droughts_220pc_1med_severity'] = kstest_apply(ds_model.droughts_220pc_1med_severity, awap.droughts_220pc_1med_severity)

    return ds_sig

# --- mannwhitneyu

def check(list):
    # check if elements in list are identical
    return all(i == list[0] for i in list)

def mannwhitneyu(x,y):
    # drop nans
    xx = x[~np.isnan(x)]
    yy = y[~np.isnan(y)]
    # check if array is 0 len
    if (len(xx) == 0) and (len(yy) == 0):
        # both were full of nans, set output to nan
        p_value = np.nan
    elif (len(xx) == 0) or (len(yy) == 0):
        # one is a nan, try the test. Most likely will fail?
        # tau, p_value = stats.mannwhitneyu(xx, yy)
        p_value = np.nan
    else:
        # checking if elements in list are identical
        check_xx = check(xx)
        check_yy = check(yy)

        if (check_xx == True) and (check_yy == True) and xx[0] == yy[0]:
            p_value = 1 # identical arrays, accept null
        else:
            tau, p_value = stats.mannwhitneyu(xx, yy)
    return p_value


def mannwhitneyu_apply(x,y,dim='year'):
    # dim name is the name of the dimension we're applying it over
    return xr.apply_ufunc(mannwhitneyu, x , y, input_core_dims=[[dim], [dim]], vectorize=True,
        output_dtypes=[float],  join='outer' # join='outer' seems to allow unequal lengths for dims-of-interest
        )

def get_significance_droughts_mannwhitneyu(ds_model, awap):
    ds_sig = xr.Dataset()

    ds_sig['droughts_2s2e'] = mannwhitneyu_apply(ds_model.droughts_2s2e, awap.droughts_2s2e)
    ds_sig['droughts_2s2e_intensity'] = mannwhitneyu_apply(ds_model.droughts_2s2e_intensity, awap.droughts_2s2e_intensity)
    ds_sig['droughts_2s2e_severity'] = mannwhitneyu_apply(ds_model.droughts_2s2e_severity, awap.droughts_2s2e_severity)

    ds_sig['droughts_median'] = mannwhitneyu_apply(ds_model.droughts_median, awap.droughts_median)
    ds_sig['droughts_median_intensity'] = mannwhitneyu_apply(ds_model.droughts_median_intensity, awap.droughts_median_intensity)
    ds_sig['droughts_median_severity'] = mannwhitneyu_apply(ds_model.droughts_median_severity, awap.droughts_median_severity)

    ds_sig['droughts_20perc'] = mannwhitneyu_apply(ds_model.droughts_20perc, awap.droughts_20perc)
    ds_sig['droughts_20perc_intensity'] = mannwhitneyu_apply(ds_model.droughts_20perc_intensity, awap.droughts_20perc_intensity)
    ds_sig['droughts_20perc_severity'] = mannwhitneyu_apply(ds_model.droughts_20perc_severity, awap.droughts_20perc_severity)

    ds_sig['droughts_120pc_2med'] = mannwhitneyu_apply(ds_model.droughts_120pc_2med, awap.droughts_120pc_2med)
    ds_sig['droughts_120pc_2med_intensity'] = mannwhitneyu_apply(ds_model.droughts_120pc_2med_intensity, awap.droughts_120pc_2med_intensity)
    ds_sig['droughts_120pc_2med_severity'] = mannwhitneyu_apply(ds_model.droughts_120pc_2med_severity, awap.droughts_120pc_2med_severity)

    ds_sig['droughts_220pc_1med'] = mannwhitneyu_apply(ds_model.droughts_220pc_1med, awap.droughts_220pc_1med)
    ds_sig['droughts_220pc_1med_intensity'] = mannwhitneyu_apply(ds_model.droughts_220pc_1med_intensity, awap.droughts_220pc_1med_intensity)
    ds_sig['droughts_220pc_1med_severity'] = mannwhitneyu_apply(ds_model.droughts_220pc_1med_severity, awap.droughts_220pc_1med_severity)

    return ds_sig

# ---------------------
# wilcoxon rank sum
def ranksums(x,y):
    # drop nans
    xx = x[~np.isnan(x)]
    yy = y[~np.isnan(y)]
    # check if array is 0 len
    if (len(xx) == 0) and (len(yy) == 0):
        # both were full of nans, set output to nan
        p_value = np.nan
    elif (len(xx) == 0) or (len(yy) == 0):
        # one is a nan, try the test. Most likely will fail?
        tau, p_value = stats.ranksums(xx, yy)
    else:
        # arrays have values in them, run test
        tau, p_value = stats.ranksums(xx, yy)
    return p_value

def ranksums_apply(x,y,dim='year'):
    # dim name is the name of the dimension we're applying it over
    return xr.apply_ufunc(ranksums, x , y, input_core_dims=[[dim], [dim]], vectorize=True,
        output_dtypes=[float],  join='outer' # join='outer' seems to allow unequal lengths for dims-of-interest
        )

def get_significance_droughts_ranksums(ds_model, awap):
    ds_sig = xr.Dataset()

    ds_sig['droughts_2s2e'] = ranksums_apply(ds_model.droughts_2s2e, awap.droughts_2s2e)
    ds_sig['droughts_2s2e_intensity'] = ranksums_apply(ds_model.droughts_2s2e_intensity, awap.droughts_2s2e_intensity)
    ds_sig['droughts_2s2e_severity'] = ranksums_apply(ds_model.droughts_2s2e_severity, awap.droughts_2s2e_severity)

    ds_sig['droughts_median'] = ranksums_apply(ds_model.droughts_median, awap.droughts_median)
    ds_sig['droughts_median_intensity'] = ranksums_apply(ds_model.droughts_median_intensity, awap.droughts_median_intensity)
    ds_sig['droughts_median_severity'] = ranksums_apply(ds_model.droughts_median_severity, awap.droughts_median_severity)

    ds_sig['droughts_20perc'] = ranksums_apply(ds_model.droughts_20perc, awap.droughts_20perc)
    ds_sig['droughts_20perc_intensity'] = ranksums_apply(ds_model.droughts_20perc_intensity, awap.droughts_20perc_intensity)
    ds_sig['droughts_20perc_severity'] = ranksums_apply(ds_model.droughts_20perc_severity, awap.droughts_20perc_severity)

    ds_sig['droughts_120pc_2med'] = ranksums_apply(ds_model.droughts_120pc_2med, awap.droughts_120pc_2med)
    ds_sig['droughts_120pc_2med_intensity'] = ranksums_apply(ds_model.droughts_120pc_2med_intensity, awap.droughts_120pc_2med_intensity)
    ds_sig['droughts_120pc_2med_severity'] = ranksums_apply(ds_model.droughts_120pc_2med_severity, awap.droughts_120pc_2med_severity)

    ds_sig['droughts_220pc_1med'] = ranksums_apply(ds_model.droughts_220pc_1med, awap.droughts_220pc_1med)
    ds_sig['droughts_220pc_1med_intensity'] = ranksums_apply(ds_model.droughts_220pc_1med_intensity, awap.droughts_220pc_1med_intensity)
    ds_sig['droughts_220pc_1med_severity'] = ranksums_apply(ds_model.droughts_220pc_1med_severity, awap.droughts_220pc_1med_severity)

    return ds_sig
# ---------------------
# wilcoxon - this isn't used but keeping just in case
def wilcoxon(x,y):
    # drop nans
    xx = x[~np.isnan(x)]
    yy = y[~np.isnan(y)]

    # check if array is 0 len
    if (len(xx) == 0) and (len(yy) == 0):
        # both were full of nans, set output to nan
        p_value = np.nan
    elif (len(xx) == 0) or (len(yy) == 0):
        # one is a nan, try the test. Most likely will fail?
        tau, p_value = stats.wilcoxon(xx, yy)
    else:
        # arrays have values in them, run test
        tau, p_value = stats.wilcoxon(xx, yy)
    return p_value

def wilcoxon_apply(x,y,dim='year'):
    # dim name is the name of the dimension we're applying it over
    return xr.apply_ufunc(wilcoxon, x , y, input_core_dims=[[dim], [dim]], vectorize=True,
        output_dtypes=[float],  join='outer' # join='outer' seems to allow unequal lengths for dims-of-interest
        )

def get_significance_droughts_wilcoxon(ds_model, awap):
    ds_sig = xr.Dataset()

    ds_sig['droughts_2s2e'] = wilcoxon_apply(ds_model.droughts_2s2e, awap.droughts_2s2e)
    ds_sig['droughts_2s2e_intensity'] = wilcoxon_apply(ds_model.droughts_2s2e_intensity, awap.droughts_2s2e_intensity)
    ds_sig['droughts_2s2e_severity'] = wilcoxon_apply(ds_model.droughts_2s2e_severity, awap.droughts_2s2e_severity)

    ds_sig['droughts_median'] = wilcoxon_apply(ds_model.droughts_median, awap.droughts_median)
    ds_sig['droughts_median_intensity'] = wilcoxon_apply(ds_model.droughts_median_intensity, awap.droughts_median_intensity)
    ds_sig['droughts_median_severity'] = wilcoxon_apply(ds_model.droughts_median_severity, awap.droughts_median_severity)

    ds_sig['droughts_20perc'] = wilcoxon_apply(ds_model.droughts_20perc, awap.droughts_20perc)
    ds_sig['droughts_20perc_intensity'] = wilcoxon_apply(ds_model.droughts_20perc_intensity, awap.droughts_20perc_intensity)
    ds_sig['droughts_20perc_severity'] = wilcoxon_apply(ds_model.droughts_20perc_severity, awap.droughts_20perc_severity)

    ds_sig['droughts_120pc_2med'] = wilcoxon_apply(ds_model.droughts_120pc_2med, awap.droughts_120pc_2med)
    ds_sig['droughts_120pc_2med_intensity'] = wilcoxon_apply(ds_model.droughts_120pc_2med_intensity, awap.droughts_120pc_2med_intensity)
    ds_sig['droughts_120pc_2med_severity'] = wilcoxon_apply(ds_model.droughts_120pc_2med_severity, awap.droughts_120pc_2med_severity)

    ds_sig['droughts_220pc_1med'] = wilcoxon_apply(ds_model.droughts_220pc_1med, awap.droughts_220pc_1med)
    ds_sig['droughts_220pc_1med_intensity'] = wilcoxon_apply(ds_model.droughts_220pc_1med_intensity, awap.droughts_220pc_1med_intensity)
    ds_sig['droughts_220pc_1med_severity'] = wilcoxon_apply(ds_model.droughts_220pc_1med_severity, awap.droughts_220pc_1med_severity)

    return ds_sig

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# set output dir etc
historical_year = 1900
hist_output_dir = '../files/historical_1900'
lm_threshold_startyear = 1900
lm_threshold_endyear = 2000
lm_output_dir = 'files/lastmillennium_threshold_1900-2000'

# Check if output directory exists, otherwise make it.

if not os.path.exists(hist_output_dir):
    print("... Creating %s now "  % hist_output_dir)
    os.makedirs(hist_output_dir)

if not os.path.exists(lm_output_dir):
    print("... Creating %s now "  % lm_output_dir)
    os.makedirs(lm_output_dir)


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Process CESM-LME files


# # # ---------------------------------
# print('... importing CESM-LME files')
ff1_precip_annual = import_cesmlme(filepath, '001')
ff2_precip_annual = import_cesmlme(filepath, '002')
ff3_precip_annual = import_cesmlme(filepath, '003')
ff4_precip_annual = import_cesmlme(filepath, '004')
ff5_precip_annual = import_cesmlme(filepath, '005')
ff6_precip_annual = import_cesmlme(filepath, '006')
ff7_precip_annual = import_cesmlme(filepath, '007')
ff8_precip_annual = import_cesmlme(filepath, '008')
ff9_precip_annual = import_cesmlme(filepath, '009')
ff10_precip_annual = import_cesmlme(filepath, '010')
ff11_precip_annual = import_cesmlme(filepath, '011')
ff12_precip_annual = import_cesmlme(filepath, '012')
ff13_precip_annual = import_cesmlme(filepath, '013')

# # # # ---------------------------------
# # # Process CESM-MLE files for last millennium
# #
ff1_precip_hist_annual , ff1_precip_lm_annual  = process_cesm_files(ff1_precip_annual, 'ff1', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff2_precip_hist_annual , ff2_precip_lm_annual  = process_cesm_files(ff2_precip_annual, 'ff2', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff3_precip_hist_annual , ff3_precip_lm_annual  = process_cesm_files(ff3_precip_annual, 'ff3', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff4_precip_hist_annual , ff4_precip_lm_annual  = process_cesm_files(ff4_precip_annual, 'ff4', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff5_precip_hist_annual , ff5_precip_lm_annual  = process_cesm_files(ff5_precip_annual, 'ff5', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff6_precip_hist_annual , ff6_precip_lm_annual  = process_cesm_files(ff6_precip_annual, 'ff6', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff7_precip_hist_annual , ff7_precip_lm_annual  = process_cesm_files(ff7_precip_annual, 'ff7', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff8_precip_hist_annual , ff8_precip_lm_annual  = process_cesm_files(ff8_precip_annual, 'ff8', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff9_precip_hist_annual , ff9_precip_lm_annual  = process_cesm_files(ff9_precip_annual, 'ff9', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff10_precip_hist_annual, ff10_precip_lm_annual = process_cesm_files(ff10_precip_annual, 'ff10', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff11_precip_hist_annual, ff11_precip_lm_annual = process_cesm_files(ff11_precip_annual, 'ff11', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff12_precip_hist_annual, ff12_precip_lm_annual = process_cesm_files(ff12_precip_annual, 'ff12', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
ff13_precip_hist_annual, ff13_precip_lm_annual = process_cesm_files(ff13_precip_annual, 'ff13', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

# read in processed files
ff1_precip_hist_annual  = xr.open_dataset('%s/ff1_precip_hist_annual.nc' % hist_output_dir)
ff2_precip_hist_annual  = xr.open_dataset('%s/ff2_precip_hist_annual.nc' % hist_output_dir)
ff3_precip_hist_annual  = xr.open_dataset('%s/ff3_precip_hist_annual.nc' % hist_output_dir)
ff4_precip_hist_annual  = xr.open_dataset('%s/ff4_precip_hist_annual.nc' % hist_output_dir)
ff5_precip_hist_annual  = xr.open_dataset('%s/ff5_precip_hist_annual.nc' % hist_output_dir)
ff6_precip_hist_annual  = xr.open_dataset('%s/ff6_precip_hist_annual.nc' % hist_output_dir)
ff7_precip_hist_annual  = xr.open_dataset('%s/ff7_precip_hist_annual.nc' % hist_output_dir)
ff8_precip_hist_annual  = xr.open_dataset('%s/ff8_precip_hist_annual.nc' % hist_output_dir)
ff9_precip_hist_annual  = xr.open_dataset('%s/ff9_precip_hist_annual.nc' % hist_output_dir)
ff10_precip_hist_annual = xr.open_dataset('%s/ff10_precip_hist_annual.nc' % hist_output_dir)
ff11_precip_hist_annual = xr.open_dataset('%s/ff11_precip_hist_annual.nc' % hist_output_dir)
ff12_precip_hist_annual = xr.open_dataset('%s/ff12_precip_hist_annual.nc' % hist_output_dir)
ff13_precip_hist_annual = xr.open_dataset('%s/ff13_precip_hist_annual.nc' % hist_output_dir)

# read in processed files
ff1_precip_lm_annual  = xr.open_dataset('%s/ff1_precip_lm_annual.nc' % lm_output_dir)
ff2_precip_lm_annual  = xr.open_dataset('%s/ff2_precip_lm_annual.nc' % lm_output_dir)
ff3_precip_lm_annual  = xr.open_dataset('%s/ff3_precip_lm_annual.nc' % lm_output_dir)
ff4_precip_lm_annual  = xr.open_dataset('%s/ff4_precip_lm_annual.nc' % lm_output_dir)
ff5_precip_lm_annual  = xr.open_dataset('%s/ff5_precip_lm_annual.nc' % lm_output_dir)
ff6_precip_lm_annual  = xr.open_dataset('%s/ff6_precip_lm_annual.nc' % lm_output_dir)
ff7_precip_lm_annual  = xr.open_dataset('%s/ff7_precip_lm_annual.nc' % lm_output_dir)
ff8_precip_lm_annual  = xr.open_dataset('%s/ff8_precip_lm_annual.nc' % lm_output_dir)
ff9_precip_lm_annual  = xr.open_dataset('%s/ff9_precip_lm_annual.nc' % lm_output_dir)
ff10_precip_lm_annual = xr.open_dataset('%s/ff10_precip_lm_annual.nc' % lm_output_dir)
ff11_precip_lm_annual = xr.open_dataset('%s/ff11_precip_lm_annual.nc' % lm_output_dir)
ff12_precip_lm_annual = xr.open_dataset('%s/ff12_precip_lm_annual.nc' % lm_output_dir)
ff13_precip_lm_annual = xr.open_dataset('%s/ff13_precip_lm_annual.nc' % lm_output_dir)

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Import PMIP3 files
# print('... Importing # PMIP3 files')
#
# bcc_pr = read_in_pmip3('bcc', 'pr')
# ccsm4_pr  = read_in_pmip3('ccsm4', 'pr')
# csiro_mk3l_pr  = read_in_pmip3('csiro_mk3l', 'pr')
# fgoals_gl_pr  = read_in_pmip3('fgoals_gl', 'pr')
# fgoals_s2_pr  = read_in_pmip3('fgoals_s2', 'pr')
# giss_21_pr  = read_in_pmip3('giss_21', 'pr')
# giss_22_pr  = read_in_pmip3('giss_22', 'pr')
# giss_23_pr  = read_in_pmip3('giss_23', 'pr')
# giss_24_pr  = read_in_pmip3('giss_24', 'pr')
# giss_25_pr  = read_in_pmip3('giss_25', 'pr')
# giss_26_pr  = read_in_pmip3('giss_26', 'pr')
# giss_27_pr  = read_in_pmip3('giss_27', 'pr')
# giss_28_pr  = read_in_pmip3('giss_28', 'pr')
# hadcm3_pr  = read_in_pmip3('hadcm3', 'pr')
# ipsl_pr  = read_in_pmip3('ipsl', 'pr')
# miroc_pr  = read_in_pmip3('miroc', 'pr')
# mpi_pr  = read_in_pmip3('mpi', 'pr')
# mri_pr  = read_in_pmip3('mri', 'pr')
#
# # fix some time issues
# ccsm4_pr['time'] = giss_21_pr.time
# hadcm3_pr = hadcm3_pr.resample(time='MS').mean()  # resample so we have nans were needed
# hadcm3_pr['time'] = giss_21_pr.time
#
# # #
# # # # %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # # # Process PMIP3 files
# # # # ---------------------------------
# # # # Process for the historical period
# bcc_precip_hist_annual       , bcc_precip_lm_annual        = process_pmip3_files(bcc_pr, 'bcc', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# ccsm4_precip_hist_annual     , ccsm4_precip_lm_annual      = process_pmip3_files(ccsm4_pr, 'ccsm4', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# csiro_mk3l_precip_hist_annual, csiro_mk3l_precip_lm_annual = process_pmip3_files(csiro_mk3l_pr, 'csiro_mk3l',historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# fgoals_gl_precip_hist_annual , fgoals_gl_precip_lm_annual  = process_pmip3_files(fgoals_gl_pr, 'fgoals_gl', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# fgoals_s2_precip_hist_annual , fgoals_s2_precip_lm_annual  = process_pmip3_files(fgoals_s2_pr, 'fgoals_s2', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_21_precip_hist_annual   , giss_21_precip_lm_annual    = process_pmip3_files(giss_21_pr, 'giss_21', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_22_precip_hist_annual   , giss_22_precip_lm_annual    = process_pmip3_files(giss_22_pr, 'giss_22', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_23_precip_hist_annual   , giss_23_precip_lm_annual    = process_pmip3_files(giss_23_pr, 'giss_23', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_24_precip_hist_annual   , giss_24_precip_lm_annual    = process_pmip3_files(giss_24_pr, 'giss_24', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_25_precip_hist_annual   , giss_25_precip_lm_annual    = process_pmip3_files(giss_25_pr, 'giss_25', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_26_precip_hist_annual   , giss_26_precip_lm_annual    = process_pmip3_files(giss_26_pr, 'giss_26', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_27_precip_hist_annual   , giss_27_precip_lm_annual    = process_pmip3_files(giss_27_pr, 'giss_27', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# giss_28_precip_hist_annual   , giss_28_precip_lm_annual    = process_pmip3_files(giss_28_pr, 'giss_28', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# hadcm3_precip_hist_annual    , hadcm3_precip_lm_annual     = process_pmip3_files(hadcm3_pr, 'hadcm3', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# ipsl_precip_hist_annual      , ipsl_precip_lm_annual       = process_pmip3_files(ipsl_pr, 'ipsl', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# miroc_precip_hist_annual     , miroc_precip_lm_annual      = process_pmip3_files(miroc_pr, 'miroc', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# mpi_precip_hist_annual       , mpi_precip_lm_annual        = process_pmip3_files(mpi_pr, 'mpi', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
# mri_precip_hist_annual       , mri_precip_lm_annual        = process_pmip3_files(mri_pr, 'mri', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

# read in processed files
bcc_precip_hist_annual        = xr.open_dataset('%s/bcc_precip_hist_annual.nc' % hist_output_dir)
ccsm4_precip_hist_annual      = xr.open_dataset('%s/ccsm4_precip_hist_annual.nc' % hist_output_dir)
csiro_mk3l_precip_hist_annual = xr.open_dataset('%s/csiro_mk3l_precip_hist_annual.nc' % hist_output_dir)
fgoals_gl_precip_hist_annual  = xr.open_dataset('%s/fgoals_gl_precip_hist_annual.nc' % hist_output_dir)
fgoals_s2_precip_hist_annual  = xr.open_dataset('%s/fgoals_s2_precip_hist_annual.nc' % hist_output_dir)
giss_21_precip_hist_annual    = xr.open_dataset('%s/giss_21_precip_hist_annual.nc' % hist_output_dir)
giss_22_precip_hist_annual    = xr.open_dataset('%s/giss_22_precip_hist_annual.nc' % hist_output_dir)
giss_23_precip_hist_annual    = xr.open_dataset('%s/giss_23_precip_hist_annual.nc' % hist_output_dir)
giss_24_precip_hist_annual    = xr.open_dataset('%s/giss_24_precip_hist_annual.nc' % hist_output_dir)
giss_25_precip_hist_annual    = xr.open_dataset('%s/giss_25_precip_hist_annual.nc' % hist_output_dir)
giss_26_precip_hist_annual    = xr.open_dataset('%s/giss_26_precip_hist_annual.nc' % hist_output_dir)
giss_27_precip_hist_annual    = xr.open_dataset('%s/giss_27_precip_hist_annual.nc' % hist_output_dir)
giss_28_precip_hist_annual    = xr.open_dataset('%s/giss_28_precip_hist_annual.nc' % hist_output_dir)
hadcm3_precip_hist_annual     = xr.open_dataset('%s/hadcm3_precip_hist_annual.nc' % hist_output_dir)
ipsl_precip_hist_annual       = xr.open_dataset('%s/ipsl_precip_hist_annual.nc' % hist_output_dir)
miroc_precip_hist_annual      = xr.open_dataset('%s/miroc_precip_hist_annual.nc' % hist_output_dir)
mpi_precip_hist_annual        = xr.open_dataset('%s/mpi_precip_hist_annual.nc' % hist_output_dir)
mri_precip_hist_annual        = xr.open_dataset('%s/mri_precip_hist_annual.nc' % hist_output_dir)

# read in processed files
bcc_precip_lm_annual        = xr.open_dataset('%s/bcc_precip_lm_annual.nc' % lm_output_dir)
ccsm4_precip_lm_annual      = xr.open_dataset('%s/ccsm4_precip_lm_annual.nc' % lm_output_dir)
csiro_mk3l_precip_lm_annual = xr.open_dataset('%s/csiro_mk3l_precip_lm_annual.nc' % lm_output_dir)
fgoals_gl_precip_lm_annual  = xr.open_dataset('%s/fgoals_gl_precip_lm_annual.nc' % lm_output_dir)
fgoals_s2_precip_lm_annual  = xr.open_dataset('%s/fgoals_s2_precip_lm_annual.nc' % lm_output_dir)
giss_21_precip_lm_annual    = xr.open_dataset('%s/giss_21_precip_lm_annual.nc' % lm_output_dir)
giss_22_precip_lm_annual    = xr.open_dataset('%s/giss_22_precip_lm_annual.nc' % lm_output_dir)
giss_23_precip_lm_annual    = xr.open_dataset('%s/giss_23_precip_lm_annual.nc' % lm_output_dir)
giss_24_precip_lm_annual    = xr.open_dataset('%s/giss_24_precip_lm_annual.nc' % lm_output_dir)
giss_25_precip_lm_annual    = xr.open_dataset('%s/giss_25_precip_lm_annual.nc' % lm_output_dir)
giss_26_precip_lm_annual    = xr.open_dataset('%s/giss_26_precip_lm_annual.nc' % lm_output_dir)
giss_27_precip_lm_annual    = xr.open_dataset('%s/giss_27_precip_lm_annual.nc' % lm_output_dir)
giss_28_precip_lm_annual    = xr.open_dataset('%s/giss_28_precip_lm_annual.nc' % lm_output_dir)
hadcm3_precip_lm_annual     = xr.open_dataset('%s/hadcm3_precip_lm_annual.nc'% lm_output_dir)
ipsl_precip_lm_annual       = xr.open_dataset('%s/ipsl_precip_lm_annual.nc' % lm_output_dir)
miroc_precip_lm_annual      = xr.open_dataset('%s/miroc_precip_lm_annual.nc' % lm_output_dir)
mpi_precip_lm_annual        = xr.open_dataset('%s/mpi_precip_lm_annual.nc' % lm_output_dir)
mri_precip_lm_annual        = xr.open_dataset('%s/mri_precip_lm_annual.nc' % lm_output_dir)

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------ Calculate ensemble means

# --- historical
ff_all_precip_hist_annual = xr.concat([ff1_precip_hist_annual, ff2_precip_hist_annual, ff3_precip_hist_annual, ff4_precip_hist_annual,
    ff5_precip_hist_annual, ff6_precip_hist_annual, ff7_precip_hist_annual, ff8_precip_hist_annual, ff9_precip_hist_annual,
    ff10_precip_hist_annual, ff11_precip_hist_annual, ff12_precip_hist_annual, ff13_precip_hist_annual], dim='en')

giss_all_precip_hist_annual = xr.concat([giss_21_precip_hist_annual, giss_22_precip_hist_annual, giss_23_precip_hist_annual,
    giss_24_precip_hist_annual, giss_25_precip_hist_annual, giss_26_precip_hist_annual, giss_27_precip_hist_annual,
    giss_28_precip_hist_annual], dim='en')

# --- last millennium
ff_all_precip_lm_annual = xr.concat([ff1_precip_lm_annual, ff2_precip_lm_annual, ff3_precip_lm_annual, ff4_precip_lm_annual,
    ff5_precip_lm_annual, ff6_precip_lm_annual, ff7_precip_lm_annual, ff8_precip_lm_annual, ff9_precip_lm_annual,
    ff10_precip_lm_annual, ff11_precip_lm_annual, ff12_precip_lm_annual, ff13_precip_lm_annual], dim='en')

giss_all_precip_lm_annual = xr.concat([giss_21_precip_lm_annual, giss_22_precip_lm_annual, giss_23_precip_lm_annual,
    giss_24_precip_lm_annual, giss_25_precip_lm_annual, giss_26_precip_lm_annual, giss_27_precip_lm_annual, giss_28_precip_lm_annual], dim='en')

save_netcdf_compression(ff_all_precip_hist_annual, hist_output_dir, 'ff_all_precip_hist_annual')
save_netcdf_compression(giss_all_precip_hist_annual, hist_output_dir, 'giss_all_precip_hist_annual')

save_netcdf_compression(ff_all_precip_lm_annual, lm_output_dir, 'ff_all_precip_lm_annual')
save_netcdf_compression(giss_all_precip_lm_annual, lm_output_dir, 'giss_all_precip_lm_annual')

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Subset to Australia - historical
print('... subset to Aus only')
# --- cesm-lme
ff1_precip_hist_annual_aus = get_aus(ff1_precip_hist_annual)
ff2_precip_hist_annual_aus = get_aus(ff2_precip_hist_annual)
ff3_precip_hist_annual_aus = get_aus(ff3_precip_hist_annual)
ff4_precip_hist_annual_aus = get_aus(ff4_precip_hist_annual)
ff5_precip_hist_annual_aus = get_aus(ff5_precip_hist_annual)
ff6_precip_hist_annual_aus = get_aus(ff6_precip_hist_annual)
ff7_precip_hist_annual_aus = get_aus(ff7_precip_hist_annual)
ff8_precip_hist_annual_aus = get_aus(ff8_precip_hist_annual)
ff9_precip_hist_annual_aus = get_aus(ff9_precip_hist_annual)
ff10_precip_hist_annual_aus = get_aus(ff10_precip_hist_annual)
ff11_precip_hist_annual_aus = get_aus(ff11_precip_hist_annual)
ff12_precip_hist_annual_aus = get_aus(ff12_precip_hist_annual)
ff13_precip_hist_annual_aus = get_aus(ff13_precip_hist_annual)

# --- pmip3
bcc_precip_hist_annual_aus = get_aus(bcc_precip_hist_annual)
ccsm4_precip_hist_annual_aus = get_aus(ccsm4_precip_hist_annual)
csiro_mk3l_precip_hist_annual_aus = get_aus(csiro_mk3l_precip_hist_annual)
fgoals_gl_precip_hist_annual_aus = get_aus(fgoals_gl_precip_hist_annual)
fgoals_s2_precip_hist_annual_aus = get_aus(fgoals_s2_precip_hist_annual)
giss_21_precip_hist_annual_aus = get_aus(giss_21_precip_hist_annual)
giss_22_precip_hist_annual_aus = get_aus(giss_22_precip_hist_annual)
giss_23_precip_hist_annual_aus = get_aus(giss_23_precip_hist_annual)
giss_24_precip_hist_annual_aus = get_aus(giss_24_precip_hist_annual)
giss_25_precip_hist_annual_aus = get_aus(giss_25_precip_hist_annual)
giss_26_precip_hist_annual_aus = get_aus(giss_26_precip_hist_annual)
giss_27_precip_hist_annual_aus = get_aus(giss_27_precip_hist_annual)
giss_28_precip_hist_annual_aus = get_aus(giss_28_precip_hist_annual)
hadcm3_precip_hist_annual_aus = get_aus(hadcm3_precip_hist_annual)
ipsl_precip_hist_annual_aus = get_aus(ipsl_precip_hist_annual)
miroc_precip_hist_annual_aus = get_aus(miroc_precip_hist_annual)
mpi_precip_hist_annual_aus = get_aus(mpi_precip_hist_annual)
mri_precip_hist_annual_aus = get_aus(mri_precip_hist_annual)

ff_all_precip_hist_annual_aus = get_aus(ff_all_precip_hist_annual)
giss_all_precip_hist_annual_aus = get_aus(giss_all_precip_hist_annual)


# --- Save output
save_netcdf_compression(ff1_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '1')
save_netcdf_compression(ff2_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '2')
save_netcdf_compression(ff3_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '3')
save_netcdf_compression(ff4_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '4')
save_netcdf_compression(ff5_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '5')
save_netcdf_compression(ff6_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '6')
save_netcdf_compression(ff7_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '7')
save_netcdf_compression(ff8_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '8')
save_netcdf_compression(ff9_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '9')
save_netcdf_compression(ff10_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '10')
save_netcdf_compression(ff11_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '11')
save_netcdf_compression(ff12_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '12')
save_netcdf_compression(ff13_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '13')

save_netcdf_compression(bcc_precip_hist_annual_aus, hist_output_dir, 'bcc_precip_hist_annual_aus')
save_netcdf_compression(ccsm4_precip_hist_annual_aus, hist_output_dir, 'ccsm4_precip_hist_annual_aus')
save_netcdf_compression(csiro_mk3l_precip_hist_annual_aus, hist_output_dir, 'csiro_mk3l_precip_hist_annual_aus')
save_netcdf_compression(fgoals_gl_precip_hist_annual_aus, hist_output_dir, 'fgoals_gl_precip_hist_annual_aus')
save_netcdf_compression(fgoals_s2_precip_hist_annual_aus, hist_output_dir, 'fgoals_s2_precip_hist_annual_aus')
save_netcdf_compression(giss_21_precip_hist_annual_aus, hist_output_dir, 'giss_21_precip_hist_annual_aus')
save_netcdf_compression(giss_22_precip_hist_annual_aus, hist_output_dir, 'giss_22_precip_hist_annual_aus')
save_netcdf_compression(giss_23_precip_hist_annual_aus, hist_output_dir, 'giss_23_precip_hist_annual_aus')
save_netcdf_compression(giss_24_precip_hist_annual_aus, hist_output_dir, 'giss_24_precip_hist_annual_aus')
save_netcdf_compression(giss_25_precip_hist_annual_aus, hist_output_dir, 'giss_25_precip_hist_annual_aus')
save_netcdf_compression(giss_26_precip_hist_annual_aus, hist_output_dir, 'giss_26_precip_hist_annual_aus')
save_netcdf_compression(giss_27_precip_hist_annual_aus, hist_output_dir, 'giss_27_precip_hist_annual_aus')
save_netcdf_compression(giss_28_precip_hist_annual_aus, hist_output_dir, 'giss_28_precip_hist_annual_aus')
save_netcdf_compression(hadcm3_precip_hist_annual_aus, hist_output_dir, 'hadcm3_precip_hist_annual_aus')
save_netcdf_compression(ipsl_precip_hist_annual_aus, hist_output_dir, 'ipsl_precip_hist_annual_aus')
save_netcdf_compression(miroc_precip_hist_annual_aus, hist_output_dir, 'miroc_precip_hist_annual_aus')
save_netcdf_compression(mpi_precip_hist_annual_aus, hist_output_dir, 'mpi_precip_hist_annual_aus')
save_netcdf_compression(mri_precip_hist_annual_aus, hist_output_dir, 'mri_precip_hist_annual_aus')

save_netcdf_compression(ff_all_precip_hist_annual_aus, hist_output_dir, 'ff_all_precip_hist_annual_aus')
save_netcdf_compression(giss_all_precip_hist_annual_aus, hist_output_dir, 'giss_all_precip_hist_annual_aus')


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Subset to Australia - last millennium
ff1_precip_lm_annual_aus  = get_aus(ff1_precip_lm_annual)
ff2_precip_lm_annual_aus  = get_aus(ff2_precip_lm_annual)
ff3_precip_lm_annual_aus  = get_aus(ff3_precip_lm_annual)
ff4_precip_lm_annual_aus  = get_aus(ff4_precip_lm_annual)
ff5_precip_lm_annual_aus  = get_aus(ff5_precip_lm_annual)
ff6_precip_lm_annual_aus  = get_aus(ff6_precip_lm_annual)
ff7_precip_lm_annual_aus  = get_aus(ff7_precip_lm_annual)
ff8_precip_lm_annual_aus  = get_aus(ff8_precip_lm_annual)
ff9_precip_lm_annual_aus  = get_aus(ff9_precip_lm_annual)
ff10_precip_lm_annual_aus = get_aus(ff10_precip_lm_annual)
ff11_precip_lm_annual_aus = get_aus(ff11_precip_lm_annual)
ff12_precip_lm_annual_aus = get_aus(ff12_precip_lm_annual)
ff13_precip_lm_annual_aus = get_aus(ff13_precip_lm_annual)

bcc_precip_lm_annual_aus = get_aus(bcc_precip_lm_annual )
ccsm4_precip_lm_annual_aus = get_aus(ccsm4_precip_lm_annual)
csiro_mk3l_precip_lm_annual_aus = get_aus(csiro_mk3l_precip_lm_annual)
fgoals_gl_precip_lm_annual_aus = get_aus(fgoals_gl_precip_lm_annual)
fgoals_s2_precip_lm_annual_aus = get_aus(fgoals_s2_precip_lm_annual)
giss_21_precip_lm_annual_aus = get_aus(giss_21_precip_lm_annual)
giss_22_precip_lm_annual_aus = get_aus(giss_22_precip_lm_annual)
giss_23_precip_lm_annual_aus = get_aus(giss_23_precip_lm_annual)
giss_24_precip_lm_annual_aus = get_aus(giss_24_precip_lm_annual)
giss_25_precip_lm_annual_aus = get_aus(giss_25_precip_lm_annual)
giss_26_precip_lm_annual_aus = get_aus(giss_26_precip_lm_annual)
giss_27_precip_lm_annual_aus = get_aus(giss_27_precip_lm_annual)
giss_28_precip_lm_annual_aus = get_aus(giss_28_precip_lm_annual)
hadcm3_precip_lm_annual_aus = get_aus(hadcm3_precip_lm_annual)
ipsl_precip_lm_annual_aus = get_aus(ipsl_precip_lm_annual)
miroc_precip_lm_annual_aus = get_aus(miroc_precip_lm_annual)
mpi_precip_lm_annual_aus = get_aus(mpi_precip_lm_annual)
mri_precip_lm_annual_aus = get_aus(mri_precip_lm_annual)

ff_all_precip_lm_annual_aus = get_aus(ff_all_precip_lm_annual)
giss_all_precip_lm_annual_aus = get_aus(giss_all_precip_lm_annual)


# --- Save output
save_netcdf_compression(ff1_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '1')
save_netcdf_compression(ff2_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '2')
save_netcdf_compression(ff3_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '3')
save_netcdf_compression(ff4_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '4')
save_netcdf_compression(ff5_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '5')
save_netcdf_compression(ff6_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '6')
save_netcdf_compression(ff7_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '7')
save_netcdf_compression(ff8_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '8')
save_netcdf_compression(ff9_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '9')
save_netcdf_compression(ff10_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '10')
save_netcdf_compression(ff11_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '11')
save_netcdf_compression(ff12_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '12')
save_netcdf_compression(ff13_precip_hist_annual_aus, hist_output_dir, 'ff%s_precip_hist_annual_aus' % '13')

save_netcdf_compression(bcc_precip_hist_annual_aus, hist_output_dir, 'bcc_precip_hist_annual_aus')
save_netcdf_compression(ccsm4_precip_hist_annual_aus, hist_output_dir, 'ccsm4_precip_hist_annual_aus')
save_netcdf_compression(csiro_mk3l_precip_hist_annual_aus, hist_output_dir, 'csiro_mk3l_precip_hist_annual_aus')
save_netcdf_compression(fgoals_gl_precip_hist_annual_aus, hist_output_dir, 'fgoals_gl_precip_hist_annual_aus')
save_netcdf_compression(fgoals_s2_precip_hist_annual_aus, hist_output_dir, 'fgoals_s2_precip_hist_annual_aus')
save_netcdf_compression(giss_21_precip_hist_annual_aus, hist_output_dir, 'giss_21_precip_hist_annual_aus')
save_netcdf_compression(giss_22_precip_hist_annual_aus, hist_output_dir, 'giss_22_precip_hist_annual_aus')
save_netcdf_compression(giss_23_precip_hist_annual_aus, hist_output_dir, 'giss_23_precip_hist_annual_aus')
save_netcdf_compression(giss_24_precip_hist_annual_aus, hist_output_dir, 'giss_24_precip_hist_annual_aus')
save_netcdf_compression(giss_25_precip_hist_annual_aus, hist_output_dir, 'giss_25_precip_hist_annual_aus')
save_netcdf_compression(giss_26_precip_hist_annual_aus, hist_output_dir, 'giss_26_precip_hist_annual_aus')
save_netcdf_compression(giss_27_precip_hist_annual_aus, hist_output_dir, 'giss_27_precip_hist_annual_aus')
save_netcdf_compression(giss_28_precip_hist_annual_aus, hist_output_dir, 'giss_28_precip_hist_annual_aus')
save_netcdf_compression(hadcm3_precip_hist_annual_aus, hist_output_dir, 'hadcm3_precip_hist_annual_aus')
save_netcdf_compression(ipsl_precip_hist_annual_aus, hist_output_dir, 'ipsl_precip_hist_annual_aus')
save_netcdf_compression(miroc_precip_hist_annual_aus, hist_output_dir, 'miroc_precip_hist_annual_aus')
save_netcdf_compression(mpi_precip_hist_annual_aus, hist_output_dir, 'mpi_precip_hist_annual_aus')
save_netcdf_compression(mri_precip_hist_annual_aus, hist_output_dir, 'mri_precip_hist_annual_aus')


save_netcdf_compression(ff1_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '1')
save_netcdf_compression(ff2_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '2')
save_netcdf_compression(ff3_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '3')
save_netcdf_compression(ff4_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '4')
save_netcdf_compression(ff5_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '5')
save_netcdf_compression(ff6_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '6')
save_netcdf_compression(ff7_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '7')
save_netcdf_compression(ff8_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '8')
save_netcdf_compression(ff9_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '9')
save_netcdf_compression(ff10_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '10')
save_netcdf_compression(ff11_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '11')
save_netcdf_compression(ff12_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '12')
save_netcdf_compression(ff13_precip_lm_annual_aus, lm_output_dir, 'ff%s_precip_lm_annual_aus' % '13')

save_netcdf_compression(bcc_precip_lm_annual_aus, lm_output_dir, 'bcc_precip_lm_annual_aus')
save_netcdf_compression(ccsm4_precip_lm_annual_aus, lm_output_dir, 'ccsm4_precip_lm_annual_aus')
save_netcdf_compression(csiro_mk3l_precip_lm_annual_aus, lm_output_dir, 'csiro_mk3l_precip_lm_annual_aus')
save_netcdf_compression(fgoals_gl_precip_lm_annual_aus, lm_output_dir, 'fgoals_gl_precip_lm_annual_aus')
save_netcdf_compression(fgoals_s2_precip_lm_annual_aus, lm_output_dir, 'fgoals_s2_precip_lm_annual_aus')
save_netcdf_compression(giss_21_precip_lm_annual_aus, lm_output_dir, 'giss_21_precip_lm_annual_aus')
save_netcdf_compression(giss_22_precip_lm_annual_aus, lm_output_dir, 'giss_22_precip_lm_annual_aus')
save_netcdf_compression(giss_23_precip_lm_annual_aus, lm_output_dir, 'giss_23_precip_lm_annual_aus')
save_netcdf_compression(giss_24_precip_lm_annual_aus, lm_output_dir, 'giss_24_precip_lm_annual_aus')
save_netcdf_compression(giss_25_precip_lm_annual_aus, lm_output_dir, 'giss_25_precip_lm_annual_aus')
save_netcdf_compression(giss_26_precip_lm_annual_aus, lm_output_dir, 'giss_26_precip_lm_annual_aus')
save_netcdf_compression(giss_27_precip_lm_annual_aus, lm_output_dir, 'giss_27_precip_lm_annual_aus')
save_netcdf_compression(giss_28_precip_lm_annual_aus, lm_output_dir, 'giss_28_precip_lm_annual_aus')
save_netcdf_compression(hadcm3_precip_lm_annual_aus, lm_output_dir, 'hadcm3_precip_lm_annual_aus')
save_netcdf_compression(ipsl_precip_lm_annual_aus, lm_output_dir, 'ipsl_precip_lm_annual_aus')
save_netcdf_compression(miroc_precip_lm_annual_aus, lm_output_dir, 'miroc_precip_lm_annual_aus')
save_netcdf_compression(mpi_precip_lm_annual_aus, lm_output_dir, 'mpi_precip_lm_annual_aus')
save_netcdf_compression(mri_precip_lm_annual_aus, lm_output_dir, 'mri_precip_lm_annual_aus')

save_netcdf_compression(ff_all_precip_lm_annual_aus, lm_output_dir, 'ff_all_precip_lm_annual_aus')
save_netcdf_compression(giss_all_precip_lm_annual_aus, lm_output_dir, 'giss_all_precip_lm_annual_aus')

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------
# ----- AWAP
print('... awap')
awap = xr.open_dataset('/Volumes/LaCie/droughts/Monthly_total_precipitation_AWAP_1900_2016_gapfilled_with_mon_mean.nc')
awap_masked= xr.open_dataset('/Volumes/LaCie/droughts/Monthly_total_precipitation_AWAP_masked_1900_2016.nc')
awap_masked_annual = awap_masked.groupby('time.year').sum('time', skipna=False)
awap_masked_annual['PRECT_mm'] = awap_masked_annual.pre

# rename awap things to be the same as the masked version
awap = awap.rename({'longitude': 'lon', 'latitude': 'lat', 'z':'time', 'variable': 'pre'})

# replace times with those in masked
awap['time'] = awap_masked.time

# %%
awap_gf_annual = awap.groupby('time.year').sum('time', skipna=False)
awap_gf_annual['PRECT_mm'] = awap_gf_annual.pre

awap_gf_annual = droughts_historical_fromyear(awap_gf_annual, historical_year, hist_output_dir, 'awap_gf')

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------
#  regridding
awap_masked_annual_bcc_res = regrid_files(awap_gf_annual, bcc_precip_hist_annual_aus)
awap_masked_annual_ccsm4_res = regrid_files(awap_gf_annual, ccsm4_precip_hist_annual_aus)
awap_masked_annual_csiro_mk3l_res = regrid_files(awap_gf_annual, csiro_mk3l_precip_hist_annual_aus)
awap_masked_annual_fgoals_gl_res = regrid_files(awap_gf_annual, fgoals_gl_precip_hist_annual_aus)
awap_masked_annual_fgoals_s2_res = regrid_files(awap_gf_annual, fgoals_s2_precip_hist_annual_aus)
awap_masked_annual_giss_21_res = regrid_files(awap_gf_annual, giss_21_precip_hist_annual_aus)
awap_masked_annual_giss_22_res = regrid_files(awap_gf_annual, giss_22_precip_hist_annual_aus)
awap_masked_annual_giss_23_res = regrid_files(awap_gf_annual, giss_23_precip_hist_annual_aus)
awap_masked_annual_giss_24_res = regrid_files(awap_gf_annual, giss_24_precip_hist_annual_aus)
awap_masked_annual_giss_25_res = regrid_files(awap_gf_annual, giss_25_precip_hist_annual_aus)
awap_masked_annual_giss_26_res = regrid_files(awap_gf_annual, giss_26_precip_hist_annual_aus)
awap_masked_annual_giss_27_res = regrid_files(awap_gf_annual, giss_27_precip_hist_annual_aus)
awap_masked_annual_giss_28_res = regrid_files(awap_gf_annual, giss_28_precip_hist_annual_aus)
awap_masked_annual_hadcm3_res = regrid_files(awap_gf_annual, hadcm3_precip_hist_annual_aus)
awap_masked_annual_ipsl_res = regrid_files(awap_gf_annual, ipsl_precip_hist_annual_aus)
awap_masked_annual_miroc_res = regrid_files(awap_gf_annual, miroc_precip_hist_annual_aus)
awap_masked_annual_mpi_res = regrid_files(awap_gf_annual, mpi_precip_hist_annual_aus)
awap_masked_annual_mri_res = regrid_files(awap_gf_annual, mri_precip_hist_annual_aus)
awap_masked_annual_ff1_res = regrid_files(awap_gf_annual, ff1_precip_hist_annual_aus)
awap_masked_annual_ff2_res = regrid_files(awap_gf_annual, ff2_precip_hist_annual_aus)
awap_masked_annual_ff3_res = regrid_files(awap_gf_annual, ff3_precip_hist_annual_aus)
awap_masked_annual_ff4_res = regrid_files(awap_gf_annual, ff4_precip_hist_annual_aus)
awap_masked_annual_ff5_res = regrid_files(awap_gf_annual, ff5_precip_hist_annual_aus)
awap_masked_annual_ff6_res = regrid_files(awap_gf_annual, ff6_precip_hist_annual_aus)
awap_masked_annual_ff7_res = regrid_files(awap_gf_annual, ff7_precip_hist_annual_aus)
awap_masked_annual_ff8_res = regrid_files(awap_gf_annual, ff8_precip_hist_annual_aus)
awap_masked_annual_ff9_res = regrid_files(awap_gf_annual, ff9_precip_hist_annual_aus)
awap_masked_annual_ff10_res = regrid_files(awap_gf_annual, ff10_precip_hist_annual_aus)
awap_masked_annual_ff11_res = regrid_files(awap_gf_annual, ff11_precip_hist_annual_aus)
awap_masked_annual_ff12_res = regrid_files(awap_gf_annual, ff12_precip_hist_annual_aus)
awap_masked_annual_ff13_res = regrid_files(awap_gf_annual, ff13_precip_hist_annual_aus)

awap_masked_annual_ff_all_res = regrid_files(awap_gf_annual, ff_all_precip_hist_annual_aus)
awap_masked_annual_giss_all_res = regrid_files(awap_gf_annual, giss_all_precip_hist_annual_aus)


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
print('... kstest')
# ------ KS tests
bcc_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(bcc_precip_hist_annual_aus, awap_masked_annual_bcc_res)
ccsm4_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ccsm4_precip_hist_annual_aus, awap_masked_annual_ccsm4_res)
csiro_mk3l_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(csiro_mk3l_precip_hist_annual_aus, awap_masked_annual_csiro_mk3l_res)
fgoals_gl_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(fgoals_gl_precip_hist_annual_aus, awap_masked_annual_fgoals_gl_res)
fgoals_s2_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(fgoals_s2_precip_hist_annual_aus, awap_masked_annual_fgoals_s2_res)
giss_21_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_21_precip_hist_annual_aus, awap_masked_annual_giss_21_res)
giss_22_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_22_precip_hist_annual_aus, awap_masked_annual_giss_22_res)
giss_23_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_23_precip_hist_annual_aus, awap_masked_annual_giss_23_res)
giss_24_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_24_precip_hist_annual_aus, awap_masked_annual_giss_24_res)
giss_25_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_25_precip_hist_annual_aus, awap_masked_annual_giss_25_res)
giss_26_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_26_precip_hist_annual_aus, awap_masked_annual_giss_26_res)
giss_27_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_27_precip_hist_annual_aus, awap_masked_annual_giss_27_res)
giss_28_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_28_precip_hist_annual_aus, awap_masked_annual_giss_28_res)
hadcm3_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(hadcm3_precip_hist_annual_aus, awap_masked_annual_hadcm3_res)
ipsl_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ipsl_precip_hist_annual_aus, awap_masked_annual_ipsl_res)
miroc_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(miroc_precip_hist_annual_aus, awap_masked_annual_miroc_res)
mpi_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(mpi_precip_hist_annual_aus, awap_masked_annual_mpi_res)
mri_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(mri_precip_hist_annual_aus, awap_masked_annual_mri_res)
ff1_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff1_precip_hist_annual_aus, awap_masked_annual_ff1_res)
ff2_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff2_precip_hist_annual_aus, awap_masked_annual_ff2_res)
ff3_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff3_precip_hist_annual_aus, awap_masked_annual_ff3_res)
ff4_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff4_precip_hist_annual_aus, awap_masked_annual_ff4_res)
ff5_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff5_precip_hist_annual_aus, awap_masked_annual_ff5_res)
ff6_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff6_precip_hist_annual_aus, awap_masked_annual_ff6_res)
ff7_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff7_precip_hist_annual_aus, awap_masked_annual_ff7_res)
ff8_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff8_precip_hist_annual_aus, awap_masked_annual_ff8_res)
ff9_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff9_precip_hist_annual_aus, awap_masked_annual_ff9_res)
ff10_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff10_precip_hist_annual_aus, awap_masked_annual_ff10_res)
ff11_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff11_precip_hist_annual_aus, awap_masked_annual_ff11_res)
ff12_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff12_precip_hist_annual_aus, awap_masked_annual_ff12_res)
ff13_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff13_precip_hist_annual_aus, awap_masked_annual_ff13_res)
ff_all_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(ff_all_precip_hist_annual_aus, awap_masked_annual_ff_all_res)
giss_all_hist_vs_awap_sig_kstest = get_significance_droughts_kstest(giss_all_precip_hist_annual_aus, awap_masked_annual_giss_all_res)

# --------
save_netcdf_compression(bcc_hist_vs_awap_sig_kstest, hist_output_dir, 'bcc_hist_vs_awap_sig_kstest')
save_netcdf_compression(ccsm4_hist_vs_awap_sig_kstest, hist_output_dir, 'ccsm4_hist_vs_awap_sig_kstest')
save_netcdf_compression(csiro_mk3l_hist_vs_awap_sig_kstest, hist_output_dir, 'csiro_mk3l_hist_vs_awap_sig_kstest')
save_netcdf_compression(fgoals_gl_hist_vs_awap_sig_kstest, hist_output_dir, 'fgoals_gl_hist_vs_awap_sig_kstest')
save_netcdf_compression(fgoals_s2_hist_vs_awap_sig_kstest, hist_output_dir, 'fgoals_s2_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_21_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_21_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_22_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_22_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_23_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_23_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_24_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_24_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_25_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_25_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_26_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_26_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_27_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_27_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_28_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_28_hist_vs_awap_sig_kstest')
save_netcdf_compression(hadcm3_hist_vs_awap_sig_kstest, hist_output_dir, 'hadcm3_hist_vs_awap_sig_kstest')
save_netcdf_compression(ipsl_hist_vs_awap_sig_kstest, hist_output_dir, 'ipsl_hist_vs_awap_sig_kstest')
save_netcdf_compression(miroc_hist_vs_awap_sig_kstest, hist_output_dir, 'miroc_hist_vs_awap_sig_kstest')
save_netcdf_compression(mpi_hist_vs_awap_sig_kstest, hist_output_dir, 'mpi_hist_vs_awap_sig_kstest')
save_netcdf_compression(mri_hist_vs_awap_sig_kstest, hist_output_dir, 'mri_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff1_hist_vs_awap_sig_kstest, hist_output_dir, 'ff1_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff2_hist_vs_awap_sig_kstest, hist_output_dir, 'ff2_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff3_hist_vs_awap_sig_kstest, hist_output_dir, 'ff3_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff4_hist_vs_awap_sig_kstest, hist_output_dir, 'ff4_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff5_hist_vs_awap_sig_kstest, hist_output_dir, 'ff5_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff6_hist_vs_awap_sig_kstest, hist_output_dir, 'ff6_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff7_hist_vs_awap_sig_kstest, hist_output_dir, 'ff7_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff8_hist_vs_awap_sig_kstest, hist_output_dir, 'ff8_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff9_hist_vs_awap_sig_kstest, hist_output_dir, 'ff9_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff10_hist_vs_awap_sig_kstest, hist_output_dir, 'ff10_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff11_hist_vs_awap_sig_kstest, hist_output_dir, 'ff11_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff12_hist_vs_awap_sig_kstest, hist_output_dir, 'ff12_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff13_hist_vs_awap_sig_kstest, hist_output_dir, 'ff13_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff12_hist_vs_awap_sig_kstest, hist_output_dir, 'ff12_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff13_hist_vs_awap_sig_kstest, hist_output_dir, 'ff13_hist_vs_awap_sig_kstest')
save_netcdf_compression(ff_all_hist_vs_awap_sig_kstest, hist_output_dir, 'ff_all_hist_vs_awap_sig_kstest')
save_netcdf_compression(giss_all_hist_vs_awap_sig_kstest, hist_output_dir, 'giss_all_hist_vs_awap_sig_kstest')

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---------------------
# process files
bcc_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(bcc_precip_hist_annual_aus, awap_masked_annual_bcc_res)
ccsm4_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ccsm4_precip_hist_annual_aus, awap_masked_annual_ccsm4_res)
csiro_mk3l_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(csiro_mk3l_precip_hist_annual_aus, awap_masked_annual_csiro_mk3l_res)
fgoals_gl_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(fgoals_gl_precip_hist_annual_aus, awap_masked_annual_fgoals_gl_res)
fgoals_s2_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(fgoals_s2_precip_hist_annual_aus, awap_masked_annual_fgoals_s2_res)
giss_21_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_21_precip_hist_annual_aus, awap_masked_annual_giss_21_res)
giss_22_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_22_precip_hist_annual_aus, awap_masked_annual_giss_22_res)
giss_23_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_23_precip_hist_annual_aus, awap_masked_annual_giss_23_res)
giss_24_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_24_precip_hist_annual_aus, awap_masked_annual_giss_24_res)
giss_25_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_25_precip_hist_annual_aus, awap_masked_annual_giss_25_res)
giss_26_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_26_precip_hist_annual_aus, awap_masked_annual_giss_26_res)
giss_27_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_27_precip_hist_annual_aus, awap_masked_annual_giss_27_res)
giss_28_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_28_precip_hist_annual_aus, awap_masked_annual_giss_28_res)
hadcm3_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(hadcm3_precip_hist_annual_aus, awap_masked_annual_hadcm3_res)
ipsl_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ipsl_precip_hist_annual_aus, awap_masked_annual_ipsl_res)
miroc_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(miroc_precip_hist_annual_aus, awap_masked_annual_miroc_res)
mpi_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(mpi_precip_hist_annual_aus, awap_masked_annual_mpi_res)
mri_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(mri_precip_hist_annual_aus, awap_masked_annual_mri_res)
ff1_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff1_precip_hist_annual_aus, awap_masked_annual_ff1_res)
ff2_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff2_precip_hist_annual_aus, awap_masked_annual_ff2_res)
ff3_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff3_precip_hist_annual_aus, awap_masked_annual_ff3_res)
ff4_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff4_precip_hist_annual_aus, awap_masked_annual_ff4_res)
ff5_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff5_precip_hist_annual_aus, awap_masked_annual_ff5_res)
ff6_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff6_precip_hist_annual_aus, awap_masked_annual_ff6_res)
ff7_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff7_precip_hist_annual_aus, awap_masked_annual_ff7_res)
ff8_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff8_precip_hist_annual_aus, awap_masked_annual_ff8_res)
ff9_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff9_precip_hist_annual_aus, awap_masked_annual_ff9_res)
ff10_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff10_precip_hist_annual_aus, awap_masked_annual_ff10_res)
ff11_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff11_precip_hist_annual_aus, awap_masked_annual_ff11_res)
ff12_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff12_precip_hist_annual_aus, awap_masked_annual_ff12_res)
ff13_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff13_precip_hist_annual_aus, awap_masked_annual_ff13_res)
ff_all_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff_all_precip_hist_annual_aus, awap_masked_annual_ff_all_res)
giss_all_hist_vs_awap_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_all_precip_hist_annual_aus, awap_masked_annual_giss_all_res)

# ---------------------
save_netcdf_compression(bcc_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'bcc_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ccsm4_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ccsm4_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(csiro_mk3l_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'csiro_mk3l_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(fgoals_gl_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'fgoals_gl_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(fgoals_s2_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'fgoals_s2_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_21_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_21_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_22_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_22_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_23_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_23_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_24_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_24_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_25_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_25_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_26_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_26_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_27_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_27_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_28_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_28_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(hadcm3_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'hadcm3_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ipsl_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ipsl_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(miroc_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'miroc_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(mpi_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'mpi_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(mri_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'mri_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff1_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff1_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff2_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff2_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff3_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff3_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff4_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff4_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff5_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff5_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff6_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff6_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff7_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff7_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff8_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff8_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff9_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff9_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff10_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff10_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff11_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff11_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff12_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff12_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff13_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff13_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(ff_all_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'ff_all_hist_vs_awap_sig_mannwhitneyu')
save_netcdf_compression(giss_all_hist_vs_awap_sig_mannwhitneyu, hist_output_dir, 'giss_all_hist_vs_awap_sig_mannwhitneyu')



# bcc_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(bcc_precip_hist_annual_aus, awap_masked_annual_bcc_res)
# ccsm4_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ccsm4_precip_hist_annual_aus, awap_masked_annual_ccsm4_res)
# csiro_mk3l_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(csiro_mk3l_precip_hist_annual_aus, awap_masked_annual_csiro_mk3l_res)
# fgoals_gl_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(fgoals_gl_precip_hist_annual_aus, awap_masked_annual_fgoals_gl_res)
# fgoals_s2_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(fgoals_s2_precip_hist_annual_aus, awap_masked_annual_fgoals_s2_res)
# giss_21_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_21_precip_hist_annual_aus, awap_masked_annual_giss_21_res)
# giss_22_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_22_precip_hist_annual_aus, awap_masked_annual_giss_22_res)
# giss_23_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_23_precip_hist_annual_aus, awap_masked_annual_giss_23_res)
# giss_24_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_24_precip_hist_annual_aus, awap_masked_annual_giss_24_res)
# giss_25_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_25_precip_hist_annual_aus, awap_masked_annual_giss_25_res)
# giss_26_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_26_precip_hist_annual_aus, awap_masked_annual_giss_26_res)
# giss_27_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_27_precip_hist_annual_aus, awap_masked_annual_giss_27_res)
# giss_28_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(giss_28_precip_hist_annual_aus, awap_masked_annual_giss_28_res)
# hadcm3_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(hadcm3_precip_hist_annual_aus, awap_masked_annual_hadcm3_res)
# ipsl_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ipsl_precip_hist_annual_aus, awap_masked_annual_ipsl_res)
# miroc_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(miroc_precip_hist_annual_aus, awap_masked_annual_miroc_res)
# mpi_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(mpi_precip_hist_annual_aus, awap_masked_annual_mpi_res)
# mri_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(mri_precip_hist_annual_aus, awap_masked_annual_mri_res)
# ff1_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff1_precip_hist_annual_aus, awap_masked_annual_ff1_res)
# ff2_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff2_precip_hist_annual_aus, awap_masked_annual_ff2_res)
# ff3_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff3_precip_hist_annual_aus, awap_masked_annual_ff3_res)
# ff4_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff4_precip_hist_annual_aus, awap_masked_annual_ff4_res)
# ff5_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff5_precip_hist_annual_aus, awap_masked_annual_ff5_res)
# ff6_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff6_precip_hist_annual_aus, awap_masked_annual_ff6_res)
# ff7_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff7_precip_hist_annual_aus, awap_masked_annual_ff7_res)
# ff8_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff8_precip_hist_annual_aus, awap_masked_annual_ff8_res)
# ff9_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff9_precip_hist_annual_aus, awap_masked_annual_ff9_res)
# ff10_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff10_precip_hist_annual_aus, awap_masked_annual_ff10_res)
# ff11_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff11_precip_hist_annual_aus, awap_masked_annual_ff11_res)
# ff12_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff12_precip_hist_annual_aus, awap_masked_annual_ff12_res)
# ff13_hist_vs_awap_sig_wilcoxon = get_significance_droughts_wilcoxon(ff13_precip_hist_annual_aus, awap_masked_annual_ff13_res)


# ---- run test
bcc_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(bcc_precip_hist_annual_aus, awap_masked_annual_bcc_res)
ccsm4_hist_vs_awap_sig_ranksums  = get_significance_droughts_ranksums(ccsm4_precip_hist_annual_aus, awap_masked_annual_ccsm4_res)
csiro_mk3l_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(csiro_mk3l_precip_hist_annual_aus, awap_masked_annual_csiro_mk3l_res)
fgoals_gl_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(fgoals_gl_precip_hist_annual_aus, awap_masked_annual_fgoals_gl_res)
fgoals_s2_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(fgoals_s2_precip_hist_annual_aus, awap_masked_annual_fgoals_s2_res)
giss_21_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_21_precip_hist_annual_aus, awap_masked_annual_giss_21_res)
giss_22_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_22_precip_hist_annual_aus, awap_masked_annual_giss_22_res)
giss_23_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_23_precip_hist_annual_aus, awap_masked_annual_giss_23_res)
giss_24_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_24_precip_hist_annual_aus, awap_masked_annual_giss_24_res)
giss_25_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_25_precip_hist_annual_aus, awap_masked_annual_giss_25_res)
giss_26_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_26_precip_hist_annual_aus, awap_masked_annual_giss_26_res)
giss_27_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_27_precip_hist_annual_aus, awap_masked_annual_giss_27_res)
giss_28_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_28_precip_hist_annual_aus, awap_masked_annual_giss_28_res)
hadcm3_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(hadcm3_precip_hist_annual_aus, awap_masked_annual_hadcm3_res)
ipsl_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ipsl_precip_hist_annual_aus, awap_masked_annual_ipsl_res)
miroc_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(miroc_precip_hist_annual_aus, awap_masked_annual_miroc_res)
mpi_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(mpi_precip_hist_annual_aus, awap_masked_annual_mpi_res)
mri_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(mri_precip_hist_annual_aus, awap_masked_annual_mri_res)
ff1_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff1_precip_hist_annual_aus, awap_masked_annual_ff1_res)
ff2_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff2_precip_hist_annual_aus, awap_masked_annual_ff2_res)
ff3_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff3_precip_hist_annual_aus, awap_masked_annual_ff3_res)
ff4_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff4_precip_hist_annual_aus, awap_masked_annual_ff4_res)
ff5_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff5_precip_hist_annual_aus, awap_masked_annual_ff5_res)
ff6_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff6_precip_hist_annual_aus, awap_masked_annual_ff6_res)
ff7_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff7_precip_hist_annual_aus, awap_masked_annual_ff7_res)
ff8_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff8_precip_hist_annual_aus, awap_masked_annual_ff8_res)
ff9_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff9_precip_hist_annual_aus, awap_masked_annual_ff9_res)
ff10_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff10_precip_hist_annual_aus, awap_masked_annual_ff10_res)
ff11_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff11_precip_hist_annual_aus, awap_masked_annual_ff11_res)
ff12_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff12_precip_hist_annual_aus, awap_masked_annual_ff12_res)
ff13_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff13_precip_hist_annual_aus, awap_masked_annual_ff13_res)
ff_all_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(ff_all_precip_hist_annual_aus, awap_masked_annual_ff_all_res)
giss_all_hist_vs_awap_sig_ranksums = get_significance_droughts_ranksums(giss_all_precip_hist_annual_aus, awap_masked_annual_giss_all_res)

# -------
save_netcdf_compression(bcc_hist_vs_awap_sig_ranksums, hist_output_dir, 'bcc_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ccsm4_hist_vs_awap_sig_ranksums, hist_output_dir, 'ccsm4_hist_vs_awap_sig_ranksums')
save_netcdf_compression(csiro_mk3l_hist_vs_awap_sig_ranksums, hist_output_dir, 'csiro_mk3l_hist_vs_awap_sig_ranksums')
save_netcdf_compression(fgoals_gl_hist_vs_awap_sig_ranksums, hist_output_dir, 'fgoals_gl_hist_vs_awap_sig_ranksums')
save_netcdf_compression(fgoals_s2_hist_vs_awap_sig_ranksums, hist_output_dir, 'fgoals_s2_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_21_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_21_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_22_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_22_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_23_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_23_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_24_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_24_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_25_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_25_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_26_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_26_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_27_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_27_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_28_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_28_hist_vs_awap_sig_ranksums')
save_netcdf_compression(hadcm3_hist_vs_awap_sig_ranksums, hist_output_dir, 'hadcm3_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ipsl_hist_vs_awap_sig_ranksums, hist_output_dir, 'ipsl_hist_vs_awap_sig_ranksums')
save_netcdf_compression(miroc_hist_vs_awap_sig_ranksums, hist_output_dir, 'miroc_hist_vs_awap_sig_ranksums')
save_netcdf_compression(mpi_hist_vs_awap_sig_ranksums, hist_output_dir, 'mpi_hist_vs_awap_sig_ranksums')
save_netcdf_compression(mri_hist_vs_awap_sig_ranksums, hist_output_dir, 'mri_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff1_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff1_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff2_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff2_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff3_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff3_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff4_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff4_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff5_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff5_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff6_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff6_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff7_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff7_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff8_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff8_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff9_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff9_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff10_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff10_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff11_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff11_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff12_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff12_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff13_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff13_hist_vs_awap_sig_ranksums')
save_netcdf_compression(ff_all_hist_vs_awap_sig_ranksums, hist_output_dir, 'ff_all_hist_vs_awap_sig_ranksums')
save_netcdf_compression(giss_all_hist_vs_awap_sig_ranksums, hist_output_dir, 'giss_all_hist_vs_awap_sig_ranksums')

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ---- run test for hist vs lm
bcc_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(bcc_precip_hist_annual_aus, bcc_precip_lm_annual_aus)
ccsm4_hist_vs_lm_sig_kstest  = get_significance_droughts_kstest(ccsm4_precip_hist_annual_aus, ccsm4_precip_lm_annual_aus)
csiro_mk3l_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(csiro_mk3l_precip_hist_annual_aus, csiro_mk3l_precip_lm_annual_aus)
fgoals_gl_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(fgoals_gl_precip_hist_annual_aus, fgoals_gl_precip_lm_annual_aus)
fgoals_s2_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(fgoals_s2_precip_hist_annual_aus, fgoals_s2_precip_lm_annual_aus)
giss_21_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_21_precip_hist_annual_aus, giss_21_precip_lm_annual_aus)
giss_22_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_22_precip_hist_annual_aus, giss_22_precip_lm_annual_aus)
giss_23_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_23_precip_hist_annual_aus, giss_23_precip_lm_annual_aus)
giss_24_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_24_precip_hist_annual_aus, giss_24_precip_lm_annual_aus)
giss_25_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_25_precip_hist_annual_aus, giss_25_precip_lm_annual_aus)
giss_26_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_26_precip_hist_annual_aus, giss_26_precip_lm_annual_aus)
giss_27_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_27_precip_hist_annual_aus, giss_27_precip_lm_annual_aus)
giss_28_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_28_precip_hist_annual_aus, giss_28_precip_lm_annual_aus)
hadcm3_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(hadcm3_precip_hist_annual_aus, hadcm3_precip_lm_annual_aus)
ipsl_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ipsl_precip_hist_annual_aus, ipsl_precip_lm_annual_aus)
miroc_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(miroc_precip_hist_annual_aus, miroc_precip_lm_annual_aus)
mpi_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(mpi_precip_hist_annual_aus, mpi_precip_lm_annual_aus)
mri_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(mri_precip_hist_annual_aus, mri_precip_lm_annual_aus)
ff1_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff1_precip_hist_annual_aus, ff1_precip_lm_annual_aus)
ff2_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff2_precip_hist_annual_aus, ff2_precip_lm_annual_aus)
ff3_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff3_precip_hist_annual_aus, ff3_precip_lm_annual_aus)
ff4_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff4_precip_hist_annual_aus, ff4_precip_lm_annual_aus)
ff5_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff5_precip_hist_annual_aus, ff5_precip_lm_annual_aus)
ff6_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff6_precip_hist_annual_aus, ff6_precip_lm_annual_aus)
ff7_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff7_precip_hist_annual_aus, ff7_precip_lm_annual_aus)
ff8_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff8_precip_hist_annual_aus, ff8_precip_lm_annual_aus)
ff9_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff9_precip_hist_annual_aus, ff9_precip_lm_annual_aus)
ff10_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff10_precip_hist_annual_aus, ff10_precip_lm_annual_aus)
ff11_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff11_precip_hist_annual_aus, ff11_precip_lm_annual_aus)
ff12_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff12_precip_hist_annual_aus, ff12_precip_lm_annual_aus)
ff13_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff13_precip_hist_annual_aus, ff13_precip_lm_annual_aus)
ff_all_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(ff_all_precip_hist_annual_aus, ff_all_precip_lm_annual_aus)
giss_all_hist_vs_lm_sig_kstest = get_significance_droughts_kstest(giss_all_precip_hist_annual_aus, giss_all_precip_lm_annual_aus)

# -------
save_netcdf_compression(bcc_hist_vs_lm_sig_kstest, lm_output_dir, 'bcc_hist_vs_lm_sig_kstest')
save_netcdf_compression(ccsm4_hist_vs_lm_sig_kstest, lm_output_dir, 'ccsm4_hist_vs_lm_sig_kstest')
save_netcdf_compression(csiro_mk3l_hist_vs_lm_sig_kstest, lm_output_dir, 'csiro_mk3l_hist_vs_lm_sig_kstest')
save_netcdf_compression(fgoals_gl_hist_vs_lm_sig_kstest, lm_output_dir, 'fgoals_gl_hist_vs_lm_sig_kstest')
save_netcdf_compression(fgoals_s2_hist_vs_lm_sig_kstest, lm_output_dir, 'fgoals_s2_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_21_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_21_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_22_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_22_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_23_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_23_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_24_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_24_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_25_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_25_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_26_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_26_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_27_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_27_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_28_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_28_hist_vs_lm_sig_kstest')
save_netcdf_compression(hadcm3_hist_vs_lm_sig_kstest, lm_output_dir, 'hadcm3_hist_vs_lm_sig_kstest')
save_netcdf_compression(ipsl_hist_vs_lm_sig_kstest, lm_output_dir, 'ipsl_hist_vs_lm_sig_kstest')
save_netcdf_compression(miroc_hist_vs_lm_sig_kstest, lm_output_dir, 'miroc_hist_vs_lm_sig_kstest')
save_netcdf_compression(mpi_hist_vs_lm_sig_kstest, lm_output_dir, 'mpi_hist_vs_lm_sig_kstest')
save_netcdf_compression(mri_hist_vs_lm_sig_kstest, lm_output_dir, 'mri_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff1_hist_vs_lm_sig_kstest, lm_output_dir, 'ff1_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff2_hist_vs_lm_sig_kstest, lm_output_dir, 'ff2_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff3_hist_vs_lm_sig_kstest, lm_output_dir, 'ff3_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff4_hist_vs_lm_sig_kstest, lm_output_dir, 'ff4_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff5_hist_vs_lm_sig_kstest, lm_output_dir, 'ff5_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff6_hist_vs_lm_sig_kstest, lm_output_dir, 'ff6_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff7_hist_vs_lm_sig_kstest, lm_output_dir, 'ff7_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff8_hist_vs_lm_sig_kstest, lm_output_dir, 'ff8_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff9_hist_vs_lm_sig_kstest, lm_output_dir, 'ff9_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff10_hist_vs_lm_sig_kstest, lm_output_dir, 'ff10_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff11_hist_vs_lm_sig_kstest, lm_output_dir, 'ff11_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff12_hist_vs_lm_sig_kstest, lm_output_dir, 'ff12_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff13_hist_vs_lm_sig_kstest, lm_output_dir, 'ff13_hist_vs_lm_sig_kstest')
save_netcdf_compression(ff_all_hist_vs_lm_sig_kstest, lm_output_dir, 'ff_all_hist_vs_lm_sig_kstest')
save_netcdf_compression(giss_all_hist_vs_lm_sig_kstest, lm_output_dir, 'giss_all_hist_vs_lm_sig_kstest')

# ------
# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
bcc_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(bcc_precip_hist_annual_aus, bcc_precip_lm_annual_aus)
ccsm4_hist_vs_lm_sig_mannwhitneyu  = get_significance_droughts_mannwhitneyu(ccsm4_precip_hist_annual_aus, ccsm4_precip_lm_annual_aus)
csiro_mk3l_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(csiro_mk3l_precip_hist_annual_aus, csiro_mk3l_precip_lm_annual_aus)
fgoals_gl_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(fgoals_gl_precip_hist_annual_aus, fgoals_gl_precip_lm_annual_aus)
fgoals_s2_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(fgoals_s2_precip_hist_annual_aus, fgoals_s2_precip_lm_annual_aus)
giss_21_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_21_precip_hist_annual_aus, giss_21_precip_lm_annual_aus)
giss_22_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_22_precip_hist_annual_aus, giss_22_precip_lm_annual_aus)
giss_23_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_23_precip_hist_annual_aus, giss_23_precip_lm_annual_aus)
giss_24_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_24_precip_hist_annual_aus, giss_24_precip_lm_annual_aus)
giss_25_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_25_precip_hist_annual_aus, giss_25_precip_lm_annual_aus)
giss_26_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_26_precip_hist_annual_aus, giss_26_precip_lm_annual_aus)
giss_27_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_27_precip_hist_annual_aus, giss_27_precip_lm_annual_aus)
giss_28_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_28_precip_hist_annual_aus, giss_28_precip_lm_annual_aus)
hadcm3_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(hadcm3_precip_hist_annual_aus, hadcm3_precip_lm_annual_aus)
ipsl_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ipsl_precip_hist_annual_aus, ipsl_precip_lm_annual_aus)
miroc_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(miroc_precip_hist_annual_aus, miroc_precip_lm_annual_aus)
mpi_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(mpi_precip_hist_annual_aus, mpi_precip_lm_annual_aus)
mri_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(mri_precip_hist_annual_aus, mri_precip_lm_annual_aus)
ff1_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff1_precip_hist_annual_aus, ff1_precip_lm_annual_aus)
ff2_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff2_precip_hist_annual_aus, ff2_precip_lm_annual_aus)
ff3_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff3_precip_hist_annual_aus, ff3_precip_lm_annual_aus)
ff4_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff4_precip_hist_annual_aus, ff4_precip_lm_annual_aus)
ff5_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff5_precip_hist_annual_aus, ff5_precip_lm_annual_aus)
ff6_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff6_precip_hist_annual_aus, ff6_precip_lm_annual_aus)
ff7_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff7_precip_hist_annual_aus, ff7_precip_lm_annual_aus)
ff8_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff8_precip_hist_annual_aus, ff8_precip_lm_annual_aus)
ff9_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff9_precip_hist_annual_aus, ff9_precip_lm_annual_aus)
ff10_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff10_precip_hist_annual_aus, ff10_precip_lm_annual_aus)
ff11_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff11_precip_hist_annual_aus, ff11_precip_lm_annual_aus)
ff12_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff12_precip_hist_annual_aus, ff12_precip_lm_annual_aus)
ff13_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff13_precip_hist_annual_aus, ff13_precip_lm_annual_aus)
ff_all_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(ff_all_precip_hist_annual_aus, ff_all_precip_lm_annual_aus)
giss_all_hist_vs_lm_sig_mannwhitneyu = get_significance_droughts_mannwhitneyu(giss_all_precip_hist_annual_aus, giss_all_precip_lm_annual_aus)

# -------
save_netcdf_compression(bcc_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'bcc_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ccsm4_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ccsm4_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(csiro_mk3l_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'csiro_mk3l_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(fgoals_gl_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'fgoals_gl_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(fgoals_s2_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'fgoals_s2_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_21_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_21_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_22_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_22_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_23_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_23_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_24_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_24_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_25_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_25_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_26_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_26_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_27_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_27_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_28_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_28_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(hadcm3_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'hadcm3_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ipsl_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ipsl_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(miroc_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'miroc_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(mpi_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'mpi_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(mri_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'mri_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff1_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff1_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff2_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff2_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff3_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff3_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff4_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff4_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff5_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff5_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff6_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff6_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff7_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff7_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff8_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff8_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff9_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff9_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff10_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff10_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff11_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff11_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff12_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff12_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff13_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff13_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(ff_all_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'ff_all_hist_vs_lm_sig_mannwhitneyu')
save_netcdf_compression(giss_all_hist_vs_lm_sig_mannwhitneyu, lm_output_dir, 'giss_all_hist_vs_lm_sig_mannwhitneyu')


# ------
bcc_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(bcc_precip_hist_annual_aus, bcc_precip_lm_annual_aus)
ccsm4_hist_vs_lm_sig_ranksums  = get_significance_droughts_ranksums(ccsm4_precip_hist_annual_aus, ccsm4_precip_lm_annual_aus)
csiro_mk3l_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(csiro_mk3l_precip_hist_annual_aus, csiro_mk3l_precip_lm_annual_aus)
fgoals_gl_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(fgoals_gl_precip_hist_annual_aus, fgoals_gl_precip_lm_annual_aus)
fgoals_s2_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(fgoals_s2_precip_hist_annual_aus, fgoals_s2_precip_lm_annual_aus)
giss_21_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_21_precip_hist_annual_aus, giss_21_precip_lm_annual_aus)
giss_22_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_22_precip_hist_annual_aus, giss_22_precip_lm_annual_aus)
giss_23_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_23_precip_hist_annual_aus, giss_23_precip_lm_annual_aus)
giss_24_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_24_precip_hist_annual_aus, giss_24_precip_lm_annual_aus)
giss_25_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_25_precip_hist_annual_aus, giss_25_precip_lm_annual_aus)
giss_26_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_26_precip_hist_annual_aus, giss_26_precip_lm_annual_aus)
giss_27_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_27_precip_hist_annual_aus, giss_27_precip_lm_annual_aus)
giss_28_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_28_precip_hist_annual_aus, giss_28_precip_lm_annual_aus)
hadcm3_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(hadcm3_precip_hist_annual_aus, hadcm3_precip_lm_annual_aus)
ipsl_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ipsl_precip_hist_annual_aus, ipsl_precip_lm_annual_aus)
miroc_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(miroc_precip_hist_annual_aus, miroc_precip_lm_annual_aus)
mpi_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(mpi_precip_hist_annual_aus, mpi_precip_lm_annual_aus)
mri_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(mri_precip_hist_annual_aus, mri_precip_lm_annual_aus)
ff1_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff1_precip_hist_annual_aus, ff1_precip_lm_annual_aus)
ff2_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff2_precip_hist_annual_aus, ff2_precip_lm_annual_aus)
ff3_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff3_precip_hist_annual_aus, ff3_precip_lm_annual_aus)
ff4_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff4_precip_hist_annual_aus, ff4_precip_lm_annual_aus)
ff5_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff5_precip_hist_annual_aus, ff5_precip_lm_annual_aus)
ff6_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff6_precip_hist_annual_aus, ff6_precip_lm_annual_aus)
ff7_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff7_precip_hist_annual_aus, ff7_precip_lm_annual_aus)
ff8_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff8_precip_hist_annual_aus, ff8_precip_lm_annual_aus)
ff9_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff9_precip_hist_annual_aus, ff9_precip_lm_annual_aus)
ff10_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff10_precip_hist_annual_aus, ff10_precip_lm_annual_aus)
ff11_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff11_precip_hist_annual_aus, ff11_precip_lm_annual_aus)
ff12_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff12_precip_hist_annual_aus, ff12_precip_lm_annual_aus)
ff13_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff13_precip_hist_annual_aus, ff13_precip_lm_annual_aus)
ff_all_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(ff_all_precip_hist_annual_aus, ff_all_precip_lm_annual_aus)
giss_all_hist_vs_lm_sig_ranksums = get_significance_droughts_ranksums(giss_all_precip_hist_annual_aus, giss_all_precip_lm_annual_aus)

# -------
save_netcdf_compression(bcc_hist_vs_lm_sig_ranksums, lm_output_dir, 'bcc_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ccsm4_hist_vs_lm_sig_ranksums, lm_output_dir, 'ccsm4_hist_vs_lm_sig_ranksums')
save_netcdf_compression(csiro_mk3l_hist_vs_lm_sig_ranksums, lm_output_dir, 'csiro_mk3l_hist_vs_lm_sig_ranksums')
save_netcdf_compression(fgoals_gl_hist_vs_lm_sig_ranksums, lm_output_dir, 'fgoals_gl_hist_vs_lm_sig_ranksums')
save_netcdf_compression(fgoals_s2_hist_vs_lm_sig_ranksums, lm_output_dir, 'fgoals_s2_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_21_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_21_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_22_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_22_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_23_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_23_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_24_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_24_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_25_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_25_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_26_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_26_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_27_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_27_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_28_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_28_hist_vs_lm_sig_ranksums')
save_netcdf_compression(hadcm3_hist_vs_lm_sig_ranksums, lm_output_dir, 'hadcm3_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ipsl_hist_vs_lm_sig_ranksums, lm_output_dir, 'ipsl_hist_vs_lm_sig_ranksums')
save_netcdf_compression(miroc_hist_vs_lm_sig_ranksums, lm_output_dir, 'miroc_hist_vs_lm_sig_ranksums')
save_netcdf_compression(mpi_hist_vs_lm_sig_ranksums, lm_output_dir, 'mpi_hist_vs_lm_sig_ranksums')
save_netcdf_compression(mri_hist_vs_lm_sig_ranksums, lm_output_dir, 'mri_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff1_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff1_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff2_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff2_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff3_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff3_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff4_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff4_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff5_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff5_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff6_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff6_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff7_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff7_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff8_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff8_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff9_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff9_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff10_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff10_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff11_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff11_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff12_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff12_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff13_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff13_hist_vs_lm_sig_ranksums')
save_netcdf_compression(ff_all_hist_vs_lm_sig_ranksums, lm_output_dir, 'ff_all_hist_vs_lm_sig_ranksums')
save_netcdf_compression(giss_all_hist_vs_lm_sig_ranksums, lm_output_dir, 'giss_all_hist_vs_lm_sig_ranksums')


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
