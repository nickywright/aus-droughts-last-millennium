# %% Code to look at droughts in CESM-LME and PMIP3
# This script processes files for looking at droughts


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
from dask.diagnostics import ProgressBar

from scipy import stats


# %% set file path to model output
filepath = '/Volumes/LaCie/CMIP5-PMIP3/CESM-LME/mon/PRECT_v6/'
filepath_pmip3 = '/Volumes/LaCie/CMIP5-PMIP3'
filepath_cesm_mon = '/Volumes/LaCie/CMIP5-PMIP3/CESM-LME/mon'

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# some options for running
process_cesm_fullforcing_files = False
process_cesm_singleforcing_files = False

process_all_pmip3_files = False
calculate_cesm_ff_ens_means = False
calculate_giss_ens_means = False
calculate_cesm_sf_ens_means = True

subset_pmip3_hist_files_to_Aus_only = False
subset_lme_ff_hist_files_to_Aus_only = False
subset_lme_single_forcing_hist_files_to_Aus_only = True
subset_pmip3_lm_files_to_Aus_only = False
subset_lme_ff_lm_files_to_Aus_only = False
subset_lme_single_forcing_lm_files_to_Aus_only = True

process_awap = False

regrid_awap_to_model_res = False

# ---- set output directories etc
historical_year = 1900
hist_output_dir = '../files/historical_1900'

# climatology for lm files
lm_threshold_startyear = 1900
lm_threshold_endyear = 2000

lm_output_dir = '../files/lastmillennium_threshold_1900-2000'


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

def import_cesmlme_single_forcing(filepath, forcing_type, casenumber):
    casenumber_short = casenumber.lstrip('0')
    ds_precc = climate_xr_funcs.import_single_forcing_variable_cam(filepath + '/PRECC', forcing_type, casenumber, 'PRECC')
    ds_precl = climate_xr_funcs.import_single_forcing_variable_cam(filepath + '/PRECL', forcing_type, casenumber, 'PRECL')
    
    ds_precc['PRECT'] = ds_precc.PRECC + ds_precl.PRECL

    # remove so we just have PRECT
    datavars = ds_precc.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == 'PRECT': pass
        else: datavars_to_remove.append(i)
    ds_precip = ds_precc.drop(datavars_to_remove)
    
    month_length = xr.DataArray(climate_xr_funcs.get_dpm(ds_precip, calendar='noleap'), 
                                coords=[ds_precip.time], name='month_length')
    ds_precip['PRECT_mm'] = ds_precip.PRECT * 1000 * 60 * 60 * 24 * month_length
    
    ds_precip_annual = ds_precip.groupby('time.year').sum('time', skipna=False)
    ds_precip_annual.load()
    return ds_precip_annual

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
        # ds.update({'time':('time', dates['time'], attrs)})
        ds['time'] = dates['time']

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
        # ds.update({'time':('time', dates['time'], attrs)})
        ds['time'] = dates['time']

    if modelname == 'miroc':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MIROC-ESM_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MIROC-ESM_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        # ds.update({'time':('time', dates['time'], attrs)})
        ds['time'] = dates['time']

    if modelname == 'mpi':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MPI-ESM-P_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MPI-ESM-P_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        # ds.update({'time':('time', dates['time'], attrs)})
        ds['time'] = dates['time']

    if modelname == 'mri':
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MRI-CGCM3_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MRI-CGCM3_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, decode_times=False)
        # try and fix the times...
        newdates = cftime.num2date(ds_p2.time.values, 'days since 1850-01-01',  calendar='standard')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}# does this make sense?
        dates = xr.Dataset({'time': ('time', newdates, attrs)})
        # dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2['time'] = dates['time']

        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeGregorian(1851,1,1), drop=True)
        ds = xr.concat([ds_p1, ds_p2], dim='time')

        # fix times again
        new_times = cftime.date2num(ds.time, calendar='365_day', units='days since 850-01-01')
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_times, attrs)})
        dates = xr.decode_cf(dates)
        # ds.update({'time':('time', dates['time'], attrs)})
        ds['time'] = dates['time']        

    return ds

# apply drought metrics
def droughts_historical_fromyear(ds, historical_year, output_dir, model_filename):
    print('... Calculating drought metrics for %s from %s onwards' % (model_filename, historical_year))
    ds_hist = ds.where(ds['year'] >= historical_year, drop=True)    # subset to historical period
    ds_hist = ds_hist.where(ds_hist['year'] <= 2000, drop=True)  # clip to 2000 for everything
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
    ds_lm = ds_lm.where(ds_lm['year'] <= 2000, drop=True)  # clip to 2000 for everything

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
# Check if output directory exists, otherwise make it.

if not os.path.exists(hist_output_dir):
    print("... Creating %s now "  % hist_output_dir)
    os.makedirs(hist_output_dir)

if not os.path.exists(lm_output_dir):
    print("... Creating %s now "  % lm_output_dir)
    os.makedirs(lm_output_dir)

# create subfolders to try and organise things a little...
if not os.path.exists('%s/global' % hist_output_dir):
    os.makedirs('%s/global' % hist_output_dir)
if not os.path.exists('%s/aus' % hist_output_dir):
    os.makedirs('%s/aus' % hist_output_dir)
if not os.path.exists('%s/sig_tests' % hist_output_dir):
    os.makedirs('%s/sig_tests' % hist_output_dir)

if not os.path.exists('%s/global' % lm_output_dir):
    os.makedirs('%s/global' % lm_output_dir)
if not os.path.exists('%s/aus' % lm_output_dir):
    os.makedirs('%s/aus' % lm_output_dir)
if not os.path.exists('%s/sig_tests' % lm_output_dir):
    os.makedirs('%s/sig_tests' % lm_output_dir)


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Process CESM-LME files

if process_cesm_fullforcing_files is True:
    print('Processing CESM-LME full forcing files')
    # # # ---------------------------------
    print('... importing CESM-LME files')
    ff1_precip_annual = import_cesmlme(filepath, '001')
    ff1_precip_hist_annual , ff1_precip_lm_annual  = process_cesm_files(ff1_precip_annual, 'cesmlme-ff1', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff2_precip_annual = import_cesmlme(filepath, '002')
    ff2_precip_hist_annual , ff2_precip_lm_annual  = process_cesm_files(ff2_precip_annual, 'cesmlme-ff2', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff3_precip_annual = import_cesmlme(filepath, '003')
    ff3_precip_hist_annual , ff3_precip_lm_annual  = process_cesm_files(ff3_precip_annual, 'cesmlme-ff3', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff4_precip_annual = import_cesmlme(filepath, '004')
    ff4_precip_hist_annual , ff4_precip_lm_annual  = process_cesm_files(ff4_precip_annual, 'cesmlme-ff4', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff5_precip_annual = import_cesmlme(filepath, '005')
    ff5_precip_hist_annual , ff5_precip_lm_annual  = process_cesm_files(ff5_precip_annual, 'cesmlme-ff5', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff6_precip_annual = import_cesmlme(filepath, '006')
    ff6_precip_hist_annual , ff6_precip_lm_annual  = process_cesm_files(ff6_precip_annual, 'cesmlme-ff6', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff7_precip_annual = import_cesmlme(filepath, '007')
    ff7_precip_hist_annual , ff7_precip_lm_annual  = process_cesm_files(ff7_precip_annual, 'cesmlme-ff7', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff8_precip_annual = import_cesmlme(filepath, '008')
    ff8_precip_hist_annual , ff8_precip_lm_annual  = process_cesm_files(ff8_precip_annual, 'cesmlme-ff8', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff9_precip_annual = import_cesmlme(filepath, '009')
    ff9_precip_hist_annual , ff9_precip_lm_annual  = process_cesm_files(ff9_precip_annual, 'cesmlme-ff9', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff10_precip_annual = import_cesmlme(filepath, '010')
    ff10_precip_hist_annual, ff10_precip_lm_annual = process_cesm_files(ff10_precip_annual, 'cesmlme-ff10', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff11_precip_annual = import_cesmlme(filepath, '011')
    ff11_precip_hist_annual, ff11_precip_lm_annual = process_cesm_files(ff11_precip_annual, 'cesmlme-ff11', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff12_precip_annual = import_cesmlme(filepath, '012')
    ff12_precip_hist_annual, ff12_precip_lm_annual = process_cesm_files(ff12_precip_annual, 'cesmlme-ff12', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    ff13_precip_annual = import_cesmlme(filepath, '013')
    ff13_precip_hist_annual, ff13_precip_lm_annual = process_cesm_files(ff13_precip_annual, 'cesmlme-ff13', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    print('... Finished processing CESM-LME full forcing files!')
else:
    print('... Skipping initial processing of CESM-LME full forcing files')


if process_cesm_singleforcing_files is True:
    print('Processing CESM-LME single forcing files')
    # # # ---------------------------------
    ## first import, then process
    lme_850forcing3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, '850forcing', '003')
    print(lme_850forcing3_precip_annual)
    lme_850forcing3_hist_annual, lme_850forcing3_lm_annual = process_cesm_files(lme_850forcing3_precip_annual, 'cesmlme-850forcing3', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # GHG
    lme_ghg1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'GHG', '001')
    print(lme_ghg1_precip_annual)
    lme_ghg1_precip_hist_annual, lme_ghg1_precip_lm_annual = process_cesm_files(lme_ghg1_precip_annual, 'cesmlme-ghg1', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ghg2_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'GHG', '002')
    lme_ghg2_precip_hist_annual, lme_ghg2_precip_lm_annual = process_cesm_files(lme_ghg2_precip_annual, 'cesmlme-ghg2', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ghg3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'GHG', '003')
    lme_ghg3_precip_hist_annual, lme_ghg3_precip_lm_annual  = process_cesm_files(lme_ghg3_precip_annual, 'cesmlme-ghg3', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # land use
    lme_lulc1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'LULC_HurttPongratz', '001')
    lme_lulc1_precip_hist_annual, lme_lulc1_precip_lm_annual = process_cesm_files(lme_lulc1_precip_annual, 'cesmlme-lulc1', 
        historical_year, hist_output_dir+ '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_lulc2_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'LULC_HurttPongratz', '002')
    lme_lulc2_precip_hist_annual, lme_lulc2_precip_lm_annual = process_cesm_files(lme_lulc2_precip_annual, 'cesmlme-lulc2', 
        historical_year, hist_output_dir+ '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_lulc3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'LULC_HurttPongratz', '003')
    lme_lulc3_precip_hist_annual, lme_lulc3_precip_lm_annual = process_cesm_files(lme_lulc3_precip_annual, 'cesmlme-lulc3', 
        historical_year, hist_output_dir+ '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # orbital
    lme_orbital1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'ORBITAL', '001')
    lme_orbital1_precip_hist_annual, lme_orbital1_precip_lm_annual = process_cesm_files(lme_orbital1_precip_annual, 'cesmlme-orbital1', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_orbital2_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'ORBITAL', '002')
    lme_orbital2_precip_hist_annual, lme_orbital2_precip_lm_annual = process_cesm_files(lme_orbital2_precip_annual, 'cesmlme-orbital2', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_orbital3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'ORBITAL', '003')
    lme_orbital3_precip_hist_annual, lme_orbital3_precip_lm_annual = process_cesm_files(lme_orbital3_precip_annual, 'cesmlme-orbital3', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # solar
    lme_solar1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'SSI_VSK_L', '001')
    lme_solar1_precip_hist_annual, lme_solar1_precip_lm_annual  = process_cesm_files(lme_solar1_precip_annual, 'cesmlme-solar1', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_solar3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'SSI_VSK_L', '003')
    lme_solar3_precip_hist_annual, lme_solar3_precip_lm_annual  = process_cesm_files(lme_solar3_precip_annual, 'cesmlme-solar3', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_solar4_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'SSI_VSK_L', '004')
    lme_solar4_precip_hist_annual, lme_solar4_precip_lm_annual  = process_cesm_files(lme_solar4_precip_annual, 'cesmlme-solar4', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_solar5_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'SSI_VSK_L', '005')
    lme_solar5_precip_hist_annual, lme_solar5_precip_lm_annual  = process_cesm_files(lme_solar5_precip_annual, 'cesmlme-solar5', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # ozone
    lme_ozone1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'OZONE_AER', '001')
    lme_ozone1_precip_hist_annual, lme_ozone1_precip_lm_annual = process_cesm_files(lme_ozone1_precip_annual, 'cesmlme-ozone1', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ozone2_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'OZONE_AER', '002')
    lme_ozone2_precip_hist_annual, lme_ozone2_precip_lm_annual = process_cesm_files(lme_ozone2_precip_annual, 'cesmlme-ozone2', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ozone3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'OZONE_AER', '003')
    lme_ozone3_precip_hist_annual, lme_ozone3_precip_lm_annual = process_cesm_files(lme_ozone3_precip_annual, 'cesmlme-ozone3', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ozone4_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'OZONE_AER', '004')
    lme_ozone4_precip_hist_annual, lme_ozone4_precip_lm_annual = process_cesm_files(lme_ozone4_precip_annual, 'cesmlme-ozone4', 
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_ozone5_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'OZONE_AER', '005')
    lme_ozone5_precip_hist_annual, lme_ozone5_precip_lm_annual = process_cesm_files(lme_ozone5_precip_annual, 'cesmlme-ozone5',
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    # volcanoes!
    lme_volc1_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'VOLC_GRA', '001')
    lme_volc1_precip_hist_annual, lme_volc1_precip_lm_annual = process_cesm_files(lme_volc1_precip_annual, 'cesmlme-volc1',
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_volc2_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'VOLC_GRA', '002')
    lme_volc2_precip_hist_annual, lme_volc2_precip_lm_annual = process_cesm_files(lme_volc2_precip_annual, 'cesmlme-volc2',
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_volc3_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'VOLC_GRA', '003')
    lme_volc3_precip_hist_annual, lme_volc3_precip_lm_annual = process_cesm_files(lme_volc3_precip_annual, 'cesmlme-volc3',
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    lme_volc4_precip_annual = import_cesmlme_single_forcing(filepath_cesm_mon, 'VOLC_GRA', '004')
    lme_volc4_precip_hist_annual, lme_volc4_precip_lm_annual = process_cesm_files(lme_volc4_precip_annual, 'cesmlme-volc4',
        historical_year, hist_output_dir + '/global', lm_threshold_startyear, lm_threshold_endyear, lm_output_dir + '/global')

    print('... Finished processing CESM-LME single forcing files!')
else:
    print('... Skipping initial processing of CESM-LME single forcing files')

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Import PMIP3 files

if process_all_pmip3_files is True:
    print('... Importing PMIP3 files')

    bcc_pr = read_in_pmip3('bcc', 'pr')
    ccsm4_pr  = read_in_pmip3('ccsm4', 'pr')
    csiro_mk3l_pr  = read_in_pmip3('csiro_mk3l', 'pr')
    fgoals_gl_pr  = read_in_pmip3('fgoals_gl', 'pr')
    fgoals_s2_pr  = read_in_pmip3('fgoals_s2', 'pr')
    giss_21_pr  = read_in_pmip3('giss_21', 'pr')
    giss_22_pr  = read_in_pmip3('giss_22', 'pr')
    giss_23_pr  = read_in_pmip3('giss_23', 'pr')
    giss_24_pr  = read_in_pmip3('giss_24', 'pr')
    giss_25_pr  = read_in_pmip3('giss_25', 'pr')
    giss_26_pr  = read_in_pmip3('giss_26', 'pr')
    giss_27_pr  = read_in_pmip3('giss_27', 'pr')
    giss_28_pr  = read_in_pmip3('giss_28', 'pr')
    hadcm3_pr  = read_in_pmip3('hadcm3', 'pr')
    ipsl_pr  = read_in_pmip3('ipsl', 'pr')
    miroc_pr  = read_in_pmip3('miroc', 'pr')
    mpi_pr  = read_in_pmip3('mpi', 'pr')
    mri_pr  = read_in_pmip3('mri', 'pr')

    # fix some time issues
    ccsm4_pr['time'] = giss_21_pr.time
    hadcm3_pr = hadcm3_pr.resample(time='MS').mean()  # resample so we have nans were needed
    hadcm3_pr['time'] = giss_21_pr.time

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # # Process PMIP3 files
    # # # ---------------------------------
    # # # Process for the historical period
    bcc_precip_hist_annual       , bcc_precip_lm_annual        = process_pmip3_files(bcc_pr, 'bcc', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    ccsm4_precip_hist_annual     , ccsm4_precip_lm_annual      = process_pmip3_files(ccsm4_pr, 'ccsm4', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    csiro_mk3l_precip_hist_annual, csiro_mk3l_precip_lm_annual = process_pmip3_files(csiro_mk3l_pr, 'csiro_mk3l',historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    fgoals_gl_precip_hist_annual , fgoals_gl_precip_lm_annual  = process_pmip3_files(fgoals_gl_pr, 'fgoals_gl', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    fgoals_s2_precip_hist_annual , fgoals_s2_precip_lm_annual  = process_pmip3_files(fgoals_s2_pr, 'fgoals_s2', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_21_precip_hist_annual   , giss_21_precip_lm_annual    = process_pmip3_files(giss_21_pr, 'giss_21', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_22_precip_hist_annual   , giss_22_precip_lm_annual    = process_pmip3_files(giss_22_pr, 'giss_22', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_23_precip_hist_annual   , giss_23_precip_lm_annual    = process_pmip3_files(giss_23_pr, 'giss_23', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_24_precip_hist_annual   , giss_24_precip_lm_annual    = process_pmip3_files(giss_24_pr, 'giss_24', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_25_precip_hist_annual   , giss_25_precip_lm_annual    = process_pmip3_files(giss_25_pr, 'giss_25', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_26_precip_hist_annual   , giss_26_precip_lm_annual    = process_pmip3_files(giss_26_pr, 'giss_26', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_27_precip_hist_annual   , giss_27_precip_lm_annual    = process_pmip3_files(giss_27_pr, 'giss_27', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    giss_28_precip_hist_annual   , giss_28_precip_lm_annual    = process_pmip3_files(giss_28_pr, 'giss_28', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    hadcm3_precip_hist_annual    , hadcm3_precip_lm_annual     = process_pmip3_files(hadcm3_pr, 'hadcm3', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    ipsl_precip_hist_annual      , ipsl_precip_lm_annual       = process_pmip3_files(ipsl_pr, 'ipsl', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    miroc_precip_hist_annual     , miroc_precip_lm_annual      = process_pmip3_files(miroc_pr, 'miroc', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    mpi_precip_hist_annual       , mpi_precip_lm_annual        = process_pmip3_files(mpi_pr, 'mpi', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)
    mri_precip_hist_annual       , mri_precip_lm_annual        = process_pmip3_files(mri_pr, 'mri', historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir)

    print('... Finished processing PMIP3 files!')
else:
    print('... Skipping initial processing of PMIP3 files')


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------ Calculate ensemble means
if calculate_cesm_ff_ens_means is True:
    # read in processed files
    ff1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff1_precip_hist_annual.nc' % hist_output_dir)
    ff2_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff2_precip_hist_annual.nc' % hist_output_dir)
    ff3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff3_precip_hist_annual.nc' % hist_output_dir)
    ff4_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff4_precip_hist_annual.nc' % hist_output_dir)
    ff5_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff5_precip_hist_annual.nc' % hist_output_dir)
    ff6_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff6_precip_hist_annual.nc' % hist_output_dir)
    ff7_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff7_precip_hist_annual.nc' % hist_output_dir)
    ff8_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff8_precip_hist_annual.nc' % hist_output_dir)
    ff9_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff9_precip_hist_annual.nc' % hist_output_dir)
    ff10_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff10_precip_hist_annual.nc' % hist_output_dir)
    ff11_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff11_precip_hist_annual.nc' % hist_output_dir)
    ff12_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff12_precip_hist_annual.nc' % hist_output_dir)
    ff13_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff13_precip_hist_annual.nc' % hist_output_dir)

    print('Calculating CESM-LME full forcing ensemble mean')
    # --- historical
    ff_all_precip_hist_annual = xr.concat([ff1_precip_hist_annual, ff2_precip_hist_annual, ff3_precip_hist_annual, ff4_precip_hist_annual,
        ff5_precip_hist_annual, ff6_precip_hist_annual, ff7_precip_hist_annual, ff8_precip_hist_annual, ff9_precip_hist_annual,
        ff10_precip_hist_annual, ff11_precip_hist_annual, ff12_precip_hist_annual, ff13_precip_hist_annual], dim='en')
    save_netcdf_compression(ff_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-ff_all_precip_hist_annual')
    
    ff1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff1_precip_lm_annual.nc' % lm_output_dir)
    ff2_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff2_precip_lm_annual.nc' % lm_output_dir)
    ff3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff3_precip_lm_annual.nc' % lm_output_dir)
    ff4_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff4_precip_lm_annual.nc' % lm_output_dir)
    ff5_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff5_precip_lm_annual.nc' % lm_output_dir)
    ff6_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff6_precip_lm_annual.nc' % lm_output_dir)
    ff7_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff7_precip_lm_annual.nc' % lm_output_dir)
    ff8_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff8_precip_lm_annual.nc' % lm_output_dir)
    ff9_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff9_precip_lm_annual.nc' % lm_output_dir)
    ff10_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff10_precip_lm_annual.nc' % lm_output_dir)
    ff11_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff11_precip_lm_annual.nc' % lm_output_dir)
    ff12_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff12_precip_lm_annual.nc' % lm_output_dir)
    ff13_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff13_precip_lm_annual.nc' % lm_output_dir)
    # --- last millennium
    ff_all_precip_lm_annual = xr.concat([ff1_precip_lm_annual, ff2_precip_lm_annual, ff3_precip_lm_annual, ff4_precip_lm_annual,
        ff5_precip_lm_annual, ff6_precip_lm_annual, ff7_precip_lm_annual, ff8_precip_lm_annual, ff9_precip_lm_annual,
        ff10_precip_lm_annual, ff11_precip_lm_annual, ff12_precip_lm_annual, ff13_precip_lm_annual], dim='en')
    save_netcdf_compression(ff_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-ff_all_precip_lm_annual')
else:
    pass

if calculate_cesm_sf_ens_means is True:
    print('Calculating CESM-LME single forcing ensemble means')
    # --- historical
    # read in processed files
    lme_ghg1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg1_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg2_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg3_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar1_precip_hist_annual.nc' % hist_output_dir)
    lme_solar3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar4_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar4_precip_hist_annual.nc' % hist_output_dir)
    lme_solar5_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar5_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone5_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_hist_annual.nc' % hist_output_dir)
    lme_volc1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc1_precip_hist_annual.nc' % hist_output_dir)
    lme_volc2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc2_precip_hist_annual.nc' % hist_output_dir)
    lme_volc3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc3_precip_hist_annual.nc' % hist_output_dir)
    lme_volc4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc4_precip_hist_annual.nc' % hist_output_dir)

    # --- historical
    lme_ghg_all_precip_hist_annual = xr.concat([lme_ghg1_precip_hist_annual, lme_ghg2_precip_hist_annual, lme_ghg3_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_ghg_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-ghg_all_precip_hist_annual')
    lme_lulc_all_precip_hist_annual = xr.concat([lme_lulc1_precip_hist_annual, lme_lulc2_precip_hist_annual, lme_lulc3_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_lulc_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-lulc_all_precip_hist_annual')
    lme_orbital_all_precip_hist_annual = xr.concat([lme_orbital1_precip_hist_annual, lme_orbital2_precip_hist_annual, lme_orbital3_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_orbital_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-orbital_all_precip_hist_annual')
    lme_solar_all_precip_hist_annual = xr.concat([lme_solar1_precip_hist_annual, lme_solar3_precip_hist_annual, 
        lme_solar4_precip_hist_annual, lme_solar5_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_solar_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-solar_all_precip_hist_annual')
    lme_ozone_all_precip_hist_annual = xr.concat([lme_ozone1_precip_hist_annual, lme_ozone2_precip_hist_annual, 
        lme_ozone3_precip_hist_annual, lme_ozone4_precip_hist_annual, lme_ozone5_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_ozone_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-ozone_all_precip_hist_annual')
    lme_volc_all_precip_hist_annual = xr.concat([lme_volc1_precip_hist_annual, lme_volc2_precip_hist_annual, 
        lme_volc3_precip_hist_annual, lme_volc4_precip_hist_annual], dim='en')
    save_netcdf_compression(lme_volc_all_precip_hist_annual, hist_output_dir + '/global', 'cesmlme-volc_all_precip_hist_annual')

    # ------------------------
    # --- last millennium
    # read in processed files
    lme_ghg1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg1_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg2_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg3_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar1_precip_lm_annual.nc' % lm_output_dir)
    lme_solar3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar4_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar4_precip_lm_annual.nc' % lm_output_dir)
    lme_solar5_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar5_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone5_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_lm_annual.nc' % lm_output_dir)
    lme_volc1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc1_precip_lm_annual.nc' % lm_output_dir)
    lme_volc2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc2_precip_lm_annual.nc' % lm_output_dir)
    lme_volc3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc3_precip_lm_annual.nc' % lm_output_dir)
    lme_volc4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc4_precip_lm_annual.nc' % lm_output_dir)

    # --- last millennium
    lme_ghg_all_precip_lm_annual = xr.concat([lme_ghg1_precip_lm_annual, lme_ghg2_precip_lm_annual, lme_ghg3_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_ghg_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-ghg_all_precip_lm_annual')
    lme_lulc_all_precip_lm_annual = xr.concat([lme_lulc1_precip_lm_annual, lme_lulc2_precip_lm_annual, lme_lulc3_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_lulc_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-lulc_all_precip_lm_annual')
    lme_orbital_all_precip_lm_annual = xr.concat([lme_orbital1_precip_lm_annual, lme_orbital2_precip_lm_annual, lme_orbital3_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_orbital_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-orbital_all_precip_lm_annual')
    lme_solar_all_precip_lm_annual = xr.concat([lme_solar1_precip_lm_annual, lme_solar3_precip_lm_annual, 
        lme_solar4_precip_lm_annual, lme_solar5_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_solar_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-solar_all_precip_lm_annual')
    lme_ozone_all_precip_lm_annual = xr.concat([lme_ozone1_precip_lm_annual, lme_ozone2_precip_lm_annual, 
        lme_ozone3_precip_lm_annual, lme_ozone4_precip_lm_annual, lme_ozone5_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_ozone_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-ozone_all_precip_lm_annual')
    lme_volc_all_precip_lm_annual = xr.concat([lme_volc1_precip_lm_annual, lme_volc2_precip_lm_annual, 
        lme_volc3_precip_lm_annual, lme_volc4_precip_lm_annual], dim='en')
    save_netcdf_compression(lme_volc_all_precip_lm_annual, lm_output_dir + '/global', 'cesmlme-volc_all_precip_lm_annual')


else:
    pass


if calculate_giss_ens_means is True:
    print('Calculating GISS ensemble mean')
    giss_all_precip_hist_annual = xr.concat([giss_21_precip_hist_annual, giss_22_precip_hist_annual, giss_23_precip_hist_annual,
        giss_24_precip_hist_annual, giss_25_precip_hist_annual, giss_26_precip_hist_annual, giss_27_precip_hist_annual,
        giss_28_precip_hist_annual], dim='en')
    save_netcdf_compression(giss_all_precip_hist_annual, hist_output_dir, 'giss_all_precip_hist_annual')

    giss_all_precip_lm_annual = xr.concat([giss_21_precip_lm_annual, giss_22_precip_lm_annual, giss_23_precip_lm_annual,
        giss_24_precip_lm_annual, giss_25_precip_lm_annual, giss_26_precip_lm_annual, giss_27_precip_lm_annual, giss_28_precip_lm_annual], dim='en')
    save_netcdf_compression(giss_all_precip_lm_annual, lm_output_dir, 'giss_all_precip_lm_annual')
else:
    pass



# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Subset to Australia - historical
if subset_pmip3_hist_files_to_Aus_only is True:
    print('... subset historical files to Aus only')
    # -----------
    # read in processed files
    bcc_precip_hist_annual        = xr.open_dataset('%s/global/bcc_precip_hist_annual.nc' % hist_output_dir)
    ccsm4_precip_hist_annual      = xr.open_dataset('%s/global/ccsm4_precip_hist_annual.nc' % hist_output_dir)
    csiro_mk3l_precip_hist_annual = xr.open_dataset('%s/global/csiro_mk3l_precip_hist_annual.nc' % hist_output_dir)
    fgoals_gl_precip_hist_annual  = xr.open_dataset('%s/global/fgoals_gl_precip_hist_annual.nc' % hist_output_dir)
    fgoals_s2_precip_hist_annual  = xr.open_dataset('%s/global/fgoals_s2_precip_hist_annual.nc' % hist_output_dir)
    giss_21_precip_hist_annual    = xr.open_dataset('%s/global/giss_21_precip_hist_annual.nc' % hist_output_dir)
    giss_22_precip_hist_annual    = xr.open_dataset('%s/global/giss_22_precip_hist_annual.nc' % hist_output_dir)
    giss_23_precip_hist_annual    = xr.open_dataset('%s/global/giss_23_precip_hist_annual.nc' % hist_output_dir)
    giss_24_precip_hist_annual    = xr.open_dataset('%s/global/giss_24_precip_hist_annual.nc' % hist_output_dir)
    giss_25_precip_hist_annual    = xr.open_dataset('%s/global/giss_25_precip_hist_annual.nc' % hist_output_dir)
    giss_26_precip_hist_annual    = xr.open_dataset('%s/global/giss_26_precip_hist_annual.nc' % hist_output_dir)
    giss_27_precip_hist_annual    = xr.open_dataset('%s/global/giss_27_precip_hist_annual.nc' % hist_output_dir)
    giss_28_precip_hist_annual    = xr.open_dataset('%s/global/giss_28_precip_hist_annual.nc' % hist_output_dir)
    hadcm3_precip_hist_annual     = xr.open_dataset('%s/global/hadcm3_precip_hist_annual.nc' % hist_output_dir)
    ipsl_precip_hist_annual       = xr.open_dataset('%s/global/ipsl_precip_hist_annual.nc' % hist_output_dir)
    miroc_precip_hist_annual      = xr.open_dataset('%s/global/miroc_precip_hist_annual.nc' % hist_output_dir)
    mpi_precip_hist_annual        = xr.open_dataset('%s/global/mpi_precip_hist_annual.nc' % hist_output_dir)
    mri_precip_hist_annual        = xr.open_dataset('%s/global/mri_precip_hist_annual.nc' % hist_output_dir)
    giss_all_precip_hist_annual   = xr.open_dataset('%s/global/giss_all_precip_hist_annual.nc' % hist_output_dir)
    # -----------
    # subset to Australia only
    bcc_precip_hist_annual_aus = get_aus(bcc_precip_hist_annual)
    save_netcdf_compression(bcc_precip_hist_annual_aus, hist_output_dir + '/aus', 'bcc_precip_hist_annual_aus')
    ccsm4_precip_hist_annual_aus = get_aus(ccsm4_precip_hist_annual)
    save_netcdf_compression(ccsm4_precip_hist_annual_aus, hist_output_dir + '/aus', 'ccsm4_precip_hist_annual_aus')
    csiro_mk3l_precip_hist_annual_aus = get_aus(csiro_mk3l_precip_hist_annual)
    save_netcdf_compression(csiro_mk3l_precip_hist_annual_aus, hist_output_dir + '/aus', 'csiro_mk3l_precip_hist_annual_aus')
    fgoals_gl_precip_hist_annual_aus = get_aus(fgoals_gl_precip_hist_annual)
    save_netcdf_compression(fgoals_gl_precip_hist_annual_aus, hist_output_dir + '/aus', 'fgoals_gl_precip_hist_annual_aus')
    fgoals_s2_precip_hist_annual_aus = get_aus(fgoals_s2_precip_hist_annual)
    save_netcdf_compression(fgoals_s2_precip_hist_annual_aus, hist_output_dir + '/aus', 'fgoals_s2_precip_hist_annual_aus')
    giss_21_precip_hist_annual_aus = get_aus(giss_21_precip_hist_annual)
    save_netcdf_compression(giss_21_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_21_precip_hist_annual_aus')
    giss_22_precip_hist_annual_aus = get_aus(giss_22_precip_hist_annual)
    save_netcdf_compression(giss_22_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_22_precip_hist_annual_aus')
    giss_23_precip_hist_annual_aus = get_aus(giss_23_precip_hist_annual)
    save_netcdf_compression(giss_23_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_23_precip_hist_annual_aus')
    giss_24_precip_hist_annual_aus = get_aus(giss_24_precip_hist_annual)
    save_netcdf_compression(giss_24_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_24_precip_hist_annual_aus')
    giss_25_precip_hist_annual_aus = get_aus(giss_25_precip_hist_annual)
    save_netcdf_compression(giss_25_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_25_precip_hist_annual_aus')
    giss_26_precip_hist_annual_aus = get_aus(giss_26_precip_hist_annual)
    save_netcdf_compression(giss_26_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_26_precip_hist_annual_aus')
    giss_27_precip_hist_annual_aus = get_aus(giss_27_precip_hist_annual)
    save_netcdf_compression(giss_27_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_27_precip_hist_annual_aus')
    giss_28_precip_hist_annual_aus = get_aus(giss_28_precip_hist_annual)
    save_netcdf_compression(giss_28_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_28_precip_hist_annual_aus')
    hadcm3_precip_hist_annual_aus = get_aus(hadcm3_precip_hist_annual)
    save_netcdf_compression(hadcm3_precip_hist_annual_aus, hist_output_dir + '/aus', 'hadcm3_precip_hist_annual_aus')
    ipsl_precip_hist_annual_aus = get_aus(ipsl_precip_hist_annual)
    save_netcdf_compression(ipsl_precip_hist_annual_aus, hist_output_dir + '/aus', 'ipsl_precip_hist_annual_aus')
    miroc_precip_hist_annual_aus = get_aus(miroc_precip_hist_annual)
    save_netcdf_compression(miroc_precip_hist_annual_aus, hist_output_dir + '/aus', 'miroc_precip_hist_annual_aus')
    mpi_precip_hist_annual_aus = get_aus(mpi_precip_hist_annual)
    save_netcdf_compression(mpi_precip_hist_annual_aus, hist_output_dir + '/aus', 'mpi_precip_hist_annual_aus')
    mri_precip_hist_annual_aus = get_aus(mri_precip_hist_annual)
    save_netcdf_compression(mri_precip_hist_annual_aus, hist_output_dir + '/aus', 'mri_precip_hist_annual_aus')
    
    # -----------
    # ensemble mean
    giss_all_precip_hist_annual_aus = get_aus(giss_all_precip_hist_annual)
    save_netcdf_compression(giss_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_all_precip_hist_annual_aus')
else:
    pass

if subset_lme_ff_hist_files_to_Aus_only is True:

    # import files
    # read in processed files
    ff1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff1_precip_hist_annual.nc' % hist_output_dir)
    ff2_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff2_precip_hist_annual.nc' % hist_output_dir)
    ff3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff3_precip_hist_annual.nc' % hist_output_dir)
    ff4_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff4_precip_hist_annual.nc' % hist_output_dir)
    ff5_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff5_precip_hist_annual.nc' % hist_output_dir)
    ff6_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff6_precip_hist_annual.nc' % hist_output_dir)
    ff7_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff7_precip_hist_annual.nc' % hist_output_dir)
    ff8_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff8_precip_hist_annual.nc' % hist_output_dir)
    ff9_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ff9_precip_hist_annual.nc' % hist_output_dir)
    ff10_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff10_precip_hist_annual.nc' % hist_output_dir)
    ff11_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff11_precip_hist_annual.nc' % hist_output_dir)
    ff12_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff12_precip_hist_annual.nc' % hist_output_dir)
    ff13_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff13_precip_hist_annual.nc' % hist_output_dir)
    ff_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff_all_precip_hist_annual.nc' % hist_output_dir)

    # --- subset files
    ff1_precip_hist_annual_aus = get_aus(ff1_precip_hist_annual)
    save_netcdf_compression(ff1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '1')
    ff2_precip_hist_annual_aus = get_aus(ff2_precip_hist_annual)
    save_netcdf_compression(ff2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '2')
    ff3_precip_hist_annual_aus = get_aus(ff3_precip_hist_annual)
    save_netcdf_compression(ff3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '3')
    ff4_precip_hist_annual_aus = get_aus(ff4_precip_hist_annual)
    save_netcdf_compression(ff4_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '4')
    ff5_precip_hist_annual_aus = get_aus(ff5_precip_hist_annual)
    save_netcdf_compression(ff5_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '5')
    ff6_precip_hist_annual_aus = get_aus(ff6_precip_hist_annual)
    save_netcdf_compression(ff6_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '6')
    ff7_precip_hist_annual_aus = get_aus(ff7_precip_hist_annual)
    save_netcdf_compression(ff7_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '7')
    ff8_precip_hist_annual_aus = get_aus(ff8_precip_hist_annual)
    save_netcdf_compression(ff8_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '8')
    ff9_precip_hist_annual_aus = get_aus(ff9_precip_hist_annual)
    save_netcdf_compression(ff9_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '9')
    ff10_precip_hist_annual_aus = get_aus(ff10_precip_hist_annual)
    save_netcdf_compression(ff10_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '10')
    ff11_precip_hist_annual_aus = get_aus(ff11_precip_hist_annual)
    save_netcdf_compression(ff11_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '11')
    ff12_precip_hist_annual_aus = get_aus(ff12_precip_hist_annual)
    save_netcdf_compression(ff12_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '12')
    ff13_precip_hist_annual_aus = get_aus(ff13_precip_hist_annual)
    save_netcdf_compression(ff13_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff%s_precip_hist_annual_aus' % '13')
    
    # ensemble mean
    ff_all_precip_hist_annual_aus = get_aus(ff_all_precip_hist_annual)
    save_netcdf_compression(ff_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff_all_precip_hist_annual_aus')
else:
    pass


if subset_lme_single_forcing_hist_files_to_Aus_only is True:
    print('importing things.')
    # read in processed files
    lme_850forcing3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-850forcing3_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg1_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg2_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg3_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar1_precip_hist_annual.nc' % hist_output_dir)
    lme_solar3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar4_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar4_precip_hist_annual.nc' % hist_output_dir)
    lme_solar5_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-solar5_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone5_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_hist_annual.nc' % hist_output_dir)
    lme_volc1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc1_precip_hist_annual.nc' % hist_output_dir)
    lme_volc2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc2_precip_hist_annual.nc' % hist_output_dir)
    lme_volc3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc3_precip_hist_annual.nc' % hist_output_dir)
    lme_volc4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc4_precip_hist_annual.nc' % hist_output_dir)
    
    # ensemble means
    lme_ghg_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ghg_all_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc_all_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital_all_precip_hist_annual.nc' % hist_output_dir)
    lme_solar_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-solar_all_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone_all_precip_hist_annual.nc' % hist_output_dir)
    lme_volc_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-volc_all_precip_hist_annual.nc' % hist_output_dir)
    
    # --- subset files
    lme_850forcing3_precip_hist_annual_aus = get_aus(lme_850forcing3_precip_hist_annual)
    save_netcdf_compression(lme_850forcing3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-850forcing3_precip_hist_annual_aus')

    lme_ghg1_precip_hist_annual_aus = get_aus(lme_ghg1_precip_hist_annual)
    save_netcdf_compression(lme_ghg1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ghg1_precip_hist_annual_aus')
    lme_ghg2_precip_hist_annual_aus = get_aus(lme_ghg2_precip_hist_annual)
    save_netcdf_compression(lme_ghg2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ghg2_precip_hist_annual_aus')
    lme_ghg3_precip_hist_annual_aus = get_aus(lme_ghg3_precip_hist_annual)
    save_netcdf_compression(lme_ghg3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ghg3_precip_hist_annual_aus')

    lme_lulc1_precip_hist_annual_aus = get_aus(lme_lulc1_precip_hist_annual)
    save_netcdf_compression(lme_lulc1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-lulc1_precip_hist_annual_aus')
    lme_lulc2_precip_hist_annual_aus = get_aus(lme_lulc2_precip_hist_annual)
    save_netcdf_compression(lme_lulc2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-lulc2_precip_hist_annual_aus')
    lme_lulc3_precip_hist_annual_aus = get_aus(lme_lulc3_precip_hist_annual)
    save_netcdf_compression(lme_lulc3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-lulc3_precip_hist_annual_aus')

    lme_orbital1_precip_hist_annual_aus = get_aus(lme_orbital1_precip_hist_annual)
    save_netcdf_compression(lme_orbital1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-orbital1_precip_hist_annual_aus')
    lme_orbital2_precip_hist_annual_aus = get_aus(lme_orbital2_precip_hist_annual)
    save_netcdf_compression(lme_orbital2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-orbital2_precip_hist_annual_aus')
    lme_orbital3_precip_hist_annual_aus = get_aus(lme_orbital3_precip_hist_annual)
    save_netcdf_compression(lme_orbital3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-orbital3_precip_hist_annual_aus')

    lme_solar1_precip_hist_annual_aus = get_aus(lme_solar1_precip_hist_annual)
    save_netcdf_compression(lme_solar1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-solar1_precip_hist_annual_aus')
    lme_solar3_precip_hist_annual_aus = get_aus(lme_solar3_precip_hist_annual)
    save_netcdf_compression(lme_solar3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-solar3_precip_hist_annual_aus')
    lme_solar4_precip_hist_annual_aus = get_aus(lme_solar4_precip_hist_annual)
    save_netcdf_compression(lme_solar4_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-solar4_precip_hist_annual_aus')
    lme_solar5_precip_hist_annual_aus = get_aus(lme_solar5_precip_hist_annual)
    save_netcdf_compression(lme_solar5_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-solar5_precip_hist_annual_aus')

    lme_ozone1_precip_hist_annual_aus = get_aus(lme_ozone1_precip_hist_annual)
    save_netcdf_compression(lme_ozone1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone1_precip_hist_annual_aus')
    lme_ozone2_precip_hist_annual_aus = get_aus(lme_ozone2_precip_hist_annual)
    save_netcdf_compression(lme_ozone2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone2_precip_hist_annual_aus')
    lme_ozone3_precip_hist_annual_aus = get_aus(lme_ozone3_precip_hist_annual)
    save_netcdf_compression(lme_ozone3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone3_precip_hist_annual_aus')
    lme_ozone4_precip_hist_annual_aus = get_aus(lme_ozone4_precip_hist_annual)
    save_netcdf_compression(lme_ozone4_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone4_precip_hist_annual_aus')
    lme_ozone5_precip_hist_annual_aus = get_aus(lme_ozone5_precip_hist_annual)
    save_netcdf_compression(lme_ozone5_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone5_precip_hist_annual_aus')

    lme_volc1_precip_hist_annual_aus = get_aus(lme_volc1_precip_hist_annual)
    save_netcdf_compression(lme_volc1_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-volc1_precip_hist_annual_aus')
    lme_volc2_precip_hist_annual_aus = get_aus(lme_volc2_precip_hist_annual)
    save_netcdf_compression(lme_volc2_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-volc2_precip_hist_annual_aus')
    lme_volc3_precip_hist_annual_aus = get_aus(lme_volc3_precip_hist_annual)
    save_netcdf_compression(lme_volc3_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-volc3_precip_hist_annual_aus')
    lme_volc4_precip_hist_annual_aus = get_aus(lme_volc4_precip_hist_annual)
    save_netcdf_compression(lme_volc4_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-volc4_precip_hist_annual_aus')

    lme_ghg_all_precip_hist_annual_aus = get_aus(lme_ghg_all_precip_hist_annual)
    save_netcdf_compression(lme_ghg_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ghg_all_precip_hist_annual_aus')
    lme_lulc_all_precip_hist_annual_aus = get_aus(lme_lulc_all_precip_hist_annual)
    save_netcdf_compression(lme_lulc_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-lulc_all_precip_hist_annual_aus')
    lme_orbital_all_precip_hist_annual_aus = get_aus(lme_orbital_all_precip_hist_annual)
    save_netcdf_compression(lme_orbital_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-orbital_all_precip_hist_annual_aus')
    lme_solar_all_precip_hist_annual_aus = get_aus(lme_solar_all_precip_hist_annual)
    save_netcdf_compression(lme_solar_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-solar_all_precip_hist_annual_aus')
    lme_ozone_all_precip_hist_annual_aus = get_aus(lme_ozone_all_precip_hist_annual)
    save_netcdf_compression(lme_ozone_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ozone_all_precip_hist_annual_aus')
    lme_volc_all_precip_hist_annual_aus = get_aus(lme_volc_all_precip_hist_annual)
    save_netcdf_compression(lme_volc_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-volc_all_precip_hist_annual_aus')
else:
    pass


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Subset to Australia - last millennium
if subset_pmip3_lm_files_to_Aus_only is True:
    # read in processed files
    bcc_precip_lm_annual        = xr.open_dataset('%s/global/bcc_precip_lm_annual.nc' % lm_output_dir)
    ccsm4_precip_lm_annual      = xr.open_dataset('%s/global/ccsm4_precip_lm_annual.nc' % lm_output_dir)
    csiro_mk3l_precip_lm_annual = xr.open_dataset('%s/global/csiro_mk3l_precip_lm_annual.nc' % lm_output_dir)
    fgoals_gl_precip_lm_annual  = xr.open_dataset('%s/global/fgoals_gl_precip_lm_annual.nc' % lm_output_dir)
    fgoals_s2_precip_lm_annual  = xr.open_dataset('%s/global/fgoals_s2_precip_lm_annual.nc' % lm_output_dir)
    giss_21_precip_lm_annual    = xr.open_dataset('%s/global/giss_21_precip_lm_annual.nc' % lm_output_dir)
    giss_22_precip_lm_annual    = xr.open_dataset('%s/global/giss_22_precip_lm_annual.nc' % lm_output_dir)
    giss_23_precip_lm_annual    = xr.open_dataset('%s/global/giss_23_precip_lm_annual.nc' % lm_output_dir)
    giss_24_precip_lm_annual    = xr.open_dataset('%s/global/giss_24_precip_lm_annual.nc' % lm_output_dir)
    giss_25_precip_lm_annual    = xr.open_dataset('%s/global/giss_25_precip_lm_annual.nc' % lm_output_dir)
    giss_26_precip_lm_annual    = xr.open_dataset('%s/global/giss_26_precip_lm_annual.nc' % lm_output_dir)
    giss_27_precip_lm_annual    = xr.open_dataset('%s/global/giss_27_precip_lm_annual.nc' % lm_output_dir)
    giss_28_precip_lm_annual    = xr.open_dataset('%s/global/giss_28_precip_lm_annual.nc' % lm_output_dir)
    hadcm3_precip_lm_annual     = xr.open_dataset('%s/global/hadcm3_precip_lm_annual.nc'% lm_output_dir)
    ipsl_precip_lm_annual       = xr.open_dataset('%s/global/ipsl_precip_lm_annual.nc' % lm_output_dir)
    miroc_precip_lm_annual      = xr.open_dataset('%s/global/miroc_precip_lm_annual.nc' % lm_output_dir)
    mpi_precip_lm_annual        = xr.open_dataset('%s/global/mpi_precip_lm_annual.nc' % lm_output_dir)
    mri_precip_lm_annual        = xr.open_dataset('%s/global/mri_precip_lm_annual.nc' % lm_output_dir)
    giss_all_precip_lm_annual_aus = xr.open_dataset('%s/global/giss_all_precip_lm_annual.nc' % lm_output_dir)

    # subset and save file
    bcc_precip_lm_annual_aus = get_aus(bcc_precip_lm_annual)
    save_netcdf_compression(bcc_precip_lm_annual_aus, lm_output_dir + '/aus', 'bcc_precip_lm_annual_aus')
    ccsm4_precip_lm_annual_aus = get_aus(ccsm4_precip_lm_annual)
    save_netcdf_compression(ccsm4_precip_lm_annual_aus, lm_output_dir + '/aus', 'ccsm4_precip_lm_annual_aus')
    csiro_mk3l_precip_lm_annual_aus = get_aus(csiro_mk3l_precip_lm_annual)
    save_netcdf_compression(csiro_mk3l_precip_lm_annual_aus, lm_output_dir + '/aus', 'csiro_mk3l_precip_lm_annual_aus')
    fgoals_gl_precip_lm_annual_aus = get_aus(fgoals_gl_precip_lm_annual)
    save_netcdf_compression(fgoals_gl_precip_lm_annual_aus, lm_output_dir + '/aus', 'fgoals_gl_precip_lm_annual_aus')
    fgoals_s2_precip_lm_annual_aus = get_aus(fgoals_s2_precip_lm_annual)
    save_netcdf_compression(fgoals_s2_precip_lm_annual_aus, lm_output_dir + '/aus', 'fgoals_s2_precip_lm_annual_aus')
    giss_21_precip_lm_annual_aus = get_aus(giss_21_precip_lm_annual)
    save_netcdf_compression(giss_21_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_21_precip_lm_annual_aus')
    giss_22_precip_lm_annual_aus = get_aus(giss_22_precip_lm_annual)
    save_netcdf_compression(giss_22_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_22_precip_lm_annual_aus')
    giss_23_precip_lm_annual_aus = get_aus(giss_23_precip_lm_annual)
    save_netcdf_compression(giss_23_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_23_precip_lm_annual_aus')
    giss_24_precip_lm_annual_aus = get_aus(giss_24_precip_lm_annual)
    save_netcdf_compression(giss_24_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_24_precip_lm_annual_aus')
    giss_25_precip_lm_annual_aus = get_aus(giss_25_precip_lm_annual)
    save_netcdf_compression(giss_25_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_25_precip_lm_annual_aus')
    giss_26_precip_lm_annual_aus = get_aus(giss_26_precip_lm_annual)
    save_netcdf_compression(giss_26_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_26_precip_lm_annual_aus')
    giss_27_precip_lm_annual_aus = get_aus(giss_27_precip_lm_annual)
    save_netcdf_compression(giss_27_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_27_precip_lm_annual_aus')
    giss_28_precip_lm_annual_aus = get_aus(giss_28_precip_lm_annual)
    save_netcdf_compression(giss_28_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_28_precip_lm_annual_aus')
    hadcm3_precip_lm_annual_aus = get_aus(hadcm3_precip_lm_annual)
    save_netcdf_compression(hadcm3_precip_lm_annual_aus, lm_output_dir + '/aus', 'hadcm3_precip_lm_annual_aus')
    ipsl_precip_lm_annual_aus = get_aus(ipsl_precip_lm_annual)
    save_netcdf_compression(ipsl_precip_lm_annual_aus, lm_output_dir + '/aus', 'ipsl_precip_lm_annual_aus')
    miroc_precip_lm_annual_aus = get_aus(miroc_precip_lm_annual)
    save_netcdf_compression(miroc_precip_lm_annual_aus, lm_output_dir + '/aus', 'miroc_precip_lm_annual_aus')
    mpi_precip_lm_annual_aus = get_aus(mpi_precip_lm_annual)
    save_netcdf_compression(mpi_precip_lm_annual_aus, lm_output_dir + '/aus', 'mpi_precip_lm_annual_aus')
    mri_precip_lm_annual_aus = get_aus(mri_precip_lm_annual)
    save_netcdf_compression(mri_precip_lm_annual_aus, lm_output_dir + '/aus', 'mri_precip_lm_annual_aus')
    giss_all_precip_lm_annual_aus = get_aus(giss_all_precip_lm_annual)
    save_netcdf_compression(giss_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_all_precip_lm_annual_aus')
else:
    pass

if subset_lme_ff_lm_files_to_Aus_only is True:
    # read in processed files
    ff1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff1_precip_lm_annual.nc' % lm_output_dir)
    ff2_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff2_precip_lm_annual.nc' % lm_output_dir)
    ff3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff3_precip_lm_annual.nc' % lm_output_dir)
    ff4_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff4_precip_lm_annual.nc' % lm_output_dir)
    ff5_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff5_precip_lm_annual.nc' % lm_output_dir)
    ff6_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff6_precip_lm_annual.nc' % lm_output_dir)
    ff7_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff7_precip_lm_annual.nc' % lm_output_dir)
    ff8_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff8_precip_lm_annual.nc' % lm_output_dir)
    ff9_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ff9_precip_lm_annual.nc' % lm_output_dir)
    ff10_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff10_precip_lm_annual.nc' % lm_output_dir)
    ff11_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff11_precip_lm_annual.nc' % lm_output_dir)
    ff12_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff12_precip_lm_annual.nc' % lm_output_dir)
    ff13_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff13_precip_lm_annual.nc' % lm_output_dir)
    ff_aus_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff_aus_precip_lm_annual.nc' % lm_output_dir)

    print('... subset lme full forcing files to Aus only')
    ff1_precip_lm_annual_aus  = get_aus(ff1_precip_lm_annual)
    save_netcdf_compression(ff1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '1')
    ff2_precip_lm_annual_aus  = get_aus(ff2_precip_lm_annual)
    save_netcdf_compression(ff2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '2')
    ff3_precip_lm_annual_aus  = get_aus(ff3_precip_lm_annual)
    save_netcdf_compression(ff3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '3')
    ff4_precip_lm_annual_aus  = get_aus(ff4_precip_lm_annual)
    save_netcdf_compression(ff4_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '4')
    ff5_precip_lm_annual_aus  = get_aus(ff5_precip_lm_annual)
    save_netcdf_compression(ff5_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '5')
    ff6_precip_lm_annual_aus  = get_aus(ff6_precip_lm_annual)
    save_netcdf_compression(ff6_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '6')
    ff7_precip_lm_annual_aus  = get_aus(ff7_precip_lm_annual)
    save_netcdf_compression(ff7_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '7')
    ff8_precip_lm_annual_aus  = get_aus(ff8_precip_lm_annual)
    save_netcdf_compression(ff8_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '8')
    ff9_precip_lm_annual_aus  = get_aus(ff9_precip_lm_annual)
    save_netcdf_compression(ff9_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '9')
    ff10_precip_lm_annual_aus = get_aus(ff10_precip_lm_annual)
    save_netcdf_compression(ff10_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '10')
    ff11_precip_lm_annual_aus = get_aus(ff11_precip_lm_annual)
    save_netcdf_compression(ff11_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '11')
    ff12_precip_lm_annual_aus = get_aus(ff12_precip_lm_annual)
    save_netcdf_compression(ff12_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '12')
    ff13_precip_lm_annual_aus = get_aus(ff13_precip_lm_annual)
    save_netcdf_compression(ff13_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff%s_precip_lm_annual_aus' % '13')
    
    ff_all_precip_lm_annual_aus = get_aus(ff_all_precip_lm_annual)
    save_netcdf_compression(ff_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff_all_precip_lm_annual_aus')
else:
    pass

if subset_lme_single_forcing_lm_files_to_Aus_only is True:
    print('importing things.')
    # read in processed files
    lme_850forcing3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-850forcing3_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg1_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg2_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg3_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar1_precip_lm_annual.nc' % lm_output_dir)
    lme_solar3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar4_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar4_precip_lm_annual.nc' % lm_output_dir)
    lme_solar5_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-solar5_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone5_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_lm_annual.nc' % lm_output_dir)
    lme_volc1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc1_precip_lm_annual.nc' % lm_output_dir)
    lme_volc2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc2_precip_lm_annual.nc' % lm_output_dir)
    lme_volc3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc3_precip_lm_annual.nc' % lm_output_dir)
    lme_volc4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc4_precip_lm_annual.nc' % lm_output_dir)
    
    # ensemble means
    lme_ghg_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ghg_all_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc_all_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital_all_precip_lm_annual.nc' % lm_output_dir)
    lme_solar_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-solar_all_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone_all_precip_lm_annual.nc' % lm_output_dir)
    lme_volc_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-volc_all_precip_lm_annual.nc' % lm_output_dir)
    # --- subset files
    lme_850forcing3_precip_lm_annual_aus = get_aus(lme_850forcing3_precip_lm_annual)
    save_netcdf_compression(lme_850forcing3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-850forcing3_precip_lm_annual_aus')

    lme_ghg1_precip_lm_annual_aus = get_aus(lme_ghg1_precip_lm_annual)
    save_netcdf_compression(lme_ghg1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ghg1_precip_lm_annual_aus')
    lme_ghg2_precip_lm_annual_aus = get_aus(lme_ghg2_precip_lm_annual)
    save_netcdf_compression(lme_ghg2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ghg2_precip_lm_annual_aus')
    lme_ghg3_precip_lm_annual_aus = get_aus(lme_ghg3_precip_lm_annual)
    save_netcdf_compression(lme_ghg3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ghg3_precip_lm_annual_aus')

    lme_lulc1_precip_lm_annual_aus = get_aus(lme_lulc1_precip_lm_annual)
    save_netcdf_compression(lme_lulc1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-lulc1_precip_lm_annual_aus')
    lme_lulc2_precip_lm_annual_aus = get_aus(lme_lulc2_precip_lm_annual)
    save_netcdf_compression(lme_lulc2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-lulc2_precip_lm_annual_aus')
    lme_lulc3_precip_lm_annual_aus = get_aus(lme_lulc3_precip_lm_annual)
    save_netcdf_compression(lme_lulc3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-lulc3_precip_lm_annual_aus')

    lme_orbital1_precip_lm_annual_aus = get_aus(lme_orbital1_precip_lm_annual)
    save_netcdf_compression(lme_orbital1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-orbital1_precip_lm_annual_aus')
    lme_orbital2_precip_lm_annual_aus = get_aus(lme_orbital2_precip_lm_annual)
    save_netcdf_compression(lme_orbital2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-orbital2_precip_lm_annual_aus')
    lme_orbital3_precip_lm_annual_aus = get_aus(lme_orbital3_precip_lm_annual)
    save_netcdf_compression(lme_orbital3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-orbital3_precip_lm_annual_aus')

    lme_solar1_precip_lm_annual_aus = get_aus(lme_solar1_precip_lm_annual)
    save_netcdf_compression(lme_solar1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-solar1_precip_lm_annual_aus')
    lme_solar3_precip_lm_annual_aus = get_aus(lme_solar3_precip_lm_annual)
    save_netcdf_compression(lme_solar3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-solar3_precip_lm_annual_aus')
    lme_solar4_precip_lm_annual_aus = get_aus(lme_solar4_precip_lm_annual)
    save_netcdf_compression(lme_solar4_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-solar4_precip_lm_annual_aus')
    lme_solar5_precip_lm_annual_aus = get_aus(lme_solar5_precip_lm_annual)
    save_netcdf_compression(lme_solar5_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-solar5_precip_lm_annual_aus')

    lme_ozone1_precip_lm_annual_aus = get_aus(lme_ozone1_precip_lm_annual)
    save_netcdf_compression(lme_ozone1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone1_precip_lm_annual_aus')
    lme_ozone2_precip_lm_annual_aus = get_aus(lme_ozone2_precip_lm_annual)
    save_netcdf_compression(lme_ozone2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone2_precip_lm_annual_aus')
    lme_ozone3_precip_lm_annual_aus = get_aus(lme_ozone3_precip_lm_annual)
    save_netcdf_compression(lme_ozone3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone3_precip_lm_annual_aus')
    lme_ozone4_precip_lm_annual_aus = get_aus(lme_ozone4_precip_lm_annual)
    save_netcdf_compression(lme_ozone4_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone4_precip_lm_annual_aus')
    lme_ozone5_precip_lm_annual_aus = get_aus(lme_ozone5_precip_lm_annual)
    save_netcdf_compression(lme_ozone5_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone5_precip_lm_annual_aus')

    lme_volc1_precip_lm_annual_aus = get_aus(lme_volc1_precip_lm_annual)
    save_netcdf_compression(lme_volc1_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-volc1_precip_lm_annual_aus')
    lme_volc2_precip_lm_annual_aus = get_aus(lme_volc2_precip_lm_annual)
    save_netcdf_compression(lme_volc2_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-volc2_precip_lm_annual_aus')
    lme_volc3_precip_lm_annual_aus = get_aus(lme_volc3_precip_lm_annual)
    save_netcdf_compression(lme_volc3_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-volc3_precip_lm_annual_aus')
    lme_volc4_precip_lm_annual_aus = get_aus(lme_volc4_precip_lm_annual)
    save_netcdf_compression(lme_volc4_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-volc4_precip_lm_annual_aus')

    lme_ghg_all_precip_lm_annual_aus = get_aus(lme_ghg_all_precip_lm_annual)
    save_netcdf_compression(lme_ghg_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ghg_all_precip_lm_annual_aus')
    lme_lulc_all_precip_lm_annual_aus = get_aus(lme_lulc_all_precip_lm_annual)
    save_netcdf_compression(lme_lulc_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-lulc_all_precip_lm_annual_aus')
    lme_orbital_all_precip_lm_annual_aus = get_aus(lme_orbital_all_precip_lm_annual)
    save_netcdf_compression(lme_orbital_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-orbital_all_precip_lm_annual_aus')
    lme_solar_all_precip_lm_annual_aus = get_aus(lme_solar_all_precip_lm_annual)
    save_netcdf_compression(lme_solar_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-solar_all_precip_lm_annual_aus')
    lme_ozone_all_precip_lm_annual_aus = get_aus(lme_ozone_all_precip_lm_annual)
    save_netcdf_compression(lme_ozone_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ozone_all_precip_lm_annual_aus')
    lme_volc_all_precip_lm_annual_aus = get_aus(lme_volc_all_precip_lm_annual)
    save_netcdf_compression(lme_volc_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-volc_all_precip_lm_annual_aus')
else:
    pass

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---------------------
# ----- AWAP
if process_awap is True:
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

    awap_gf_annual = awap_gf_annual.where(awap_gf_annual['year'] <= 2000, drop=True)

    awap_gf_annual = droughts_historical_fromyear(awap_gf_annual, historical_year, hist_output_dir, 'awap_gf')
    save_netcdf_compression(awap_gf_annual, hist_output_dir, 'awap_gf_annual')

else:
    awap_gf_annual = xr.open_dataset('%s/awap_gf_annual.nc' % hist_output_dir)
    pass




