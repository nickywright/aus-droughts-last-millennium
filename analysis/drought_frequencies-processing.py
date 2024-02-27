# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Code to look at droughts in CESM-LME and PMIP3
# This script processes PMIP3 and CESM-LME files for looking at multi-year droughts

# Essentially, it will:
#   - Import last millennium and historical files and merge into a single time series 
#   - Calculate annual precipitation
#   - Calculate what years are in drought based on some kind of drought definition
#     (e.g. '2s2e', relative to a threshold, etc) and what your climatology is for both the
#     historical and last millenium portions
#   - Calculate a bunch of drought metrics (mean length, max length, intensity, severity...)
#   - regrid files to 2° x 2°
#   - subset regridded files to Australia
#   - save a number of netcdf files along the way

# Note: This script has been developed and added to/modified over a number of years, and so 
# there are no guarantees this whole script now works with the latest versions of all modules

# Any questions or issues? Please contact Nicky Wright (nicky.wright@sydney.edu.au)

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# --- import modules
import sys
sys.path.append('../')  # import functions to make life easier
# two python scripts from this repository
import climate_xr_funcs
import climate_droughts_xr_funcs

# other python thigns
import xarray as xr
import numpy as np
import xesmf as xe
import matplotlib.pyplot as plt
import cftime
import regionmask
import os
from dask.diagnostics import ProgressBar
from scipy import stats

# --- set file path to model output
filepath = '/Volumes/LaCie/CMIP5-PMIP3/CESM-LME/mon/PRECT_v6/'
filepath_pmip3 = '/Volumes/LaCie/CMIP5-PMIP3'
filepath_cesm_mon = '/Volumes/LaCie/CMIP5-PMIP3/CESM-LME/mon'

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# ---- set output directories etc
historical_year = 1900
hist_output_dir = '../files/historical_%s' % (historical_year)
regridded_hist_output_dir = '%s/global_2degrees' % hist_output_dir
regridded_hist_output_dir_aus = '%s/aus_2degrees' % hist_output_dir

# climatology for lm files
lm_threshold_startyear = 1900
lm_threshold_endyear = 2000

lm_output_dir = '../files/lastmillennium_threshold_%s-%s' % (lm_threshold_startyear, lm_threshold_endyear)

regridded_lm_output_dir = '%s/global_2degrees' % lm_output_dir
regridded_lm_output_dir_aus = '%s/aus_2degrees' % lm_output_dir


cntl_output_dir = '../files/control'
regridded_cntl_output_dir = '%s/global_2degrees' % cntl_output_dir
regridded_cntl_output_dir_aus = '%s/aus_2degrees' % cntl_output_dir


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Some options for running so I don't have to do everything, everytime.
# NOTE: Some have been added through time so perhaps not ALL functions run
# since they used different versions of xarray, etc...

process_cesm_fullforcing_files = False
process_cesm_singleforcing_files = False

process_all_pmip3_files = False
process_pmip3_control_files = False


calculate_cesm_ff_ens_means = False
calculate_giss_ens_means = False
calculate_cesm_sf_ens_means = False

subset_pmip3_hist_files_to_Aus_only = False
subset_lme_ff_hist_files_to_Aus_only = True
subset_lme_single_forcing_hist_files_to_Aus_only = False

subset_pmip3_lm_files_to_Aus_only = False
subset_lme_ff_lm_files_to_Aus_only = False
subset_lme_single_forcing_lm_files_to_Aus_only = False

process_awap = False

process_control_files = True
subset_control_files_to_Aus = False

regrid_awap_to_model_res = False

regrid_lme_ff_historical_files = False
regrid_lme_single_forcing_historical_files = False
regrid_pmip3_historical_files = True

regrid_lme_ff_lastmillennium_files = False
regrid_lme_single_forcing_lastmillennium_files = False
regrid_pmip3_lastmillennium_files = True

regrid_control_files = True

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Definitions

# Import CESM-LME files, subset to 1900 onwards, and calculate droughts
def import_cesmlme(filepath, casenumber):
    casenumber_short = casenumber.lstrip('0')
    ff_precip = climate_xr_funcs.import_full_forcing_precip(filepath, casenumber)
    ff_precip_annual = ff_precip.groupby('time.year').sum('time', skipna=False)
    ff_precip_annual.load()
    return ff_precip_annual

def import_cesmlme_single_forcing(filepath, forcing_type, casenumber):
    casenumber_short = casenumber.lstrip('0')
    # combine precc and precl components
    ds_precc = climate_xr_funcs.import_single_forcing_variable_cam(filepath + '/PRECC', forcing_type, casenumber, 'PRECC')
    ds_precl = climate_xr_funcs.import_single_forcing_variable_cam(filepath + '/PRECL', forcing_type, casenumber, 'PRECL')
    
    ds_precc['PRECT'] = ds_precc.PRECC + ds_precl.PRECL

    # remove so we just have PRECT
    datavars = ds_precc.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == 'PRECT': pass
        else: datavars_to_remove.append(i)
    ds_precip = ds_precc.drop_vars(datavars_to_remove)
    
    # NOTE: hardcoded calendar to be 'noleap', because that's what CESM uses
    month_length = xr.DataArray(climate_xr_funcs.get_dpm(ds_precip, calendar='noleap'), 
                                coords=[ds_precip.time], name='month_length')
    ds_precip['PRECT_mm'] = ds_precip.PRECT * 1000 * 60 * 60 * 24 * month_length
    
    ds_precip_annual = ds_precip.groupby('time.year').sum('time', skipna=False)
    ds_precip_annual.load()

    return ds_precip_annual

# ----- PMIP3 defs
# Import PMIP3 files
def read_in_pmip3(modelname, var):
    """ Read in monthly past1000 and historical netcdfs based on specified variable.
    Mergest them at the same time if both a past1000 and historical file exists.

    Outputs a monthly ds, EXCEPT for
        - ipsl
        - miroc
        - MRI
    which are returned in annual format. 

    Yep this is annoying, but only really need annual for this analysis and the calendars were really frustrating.
    One day I'll fix to be monthly output for these for other things...
    
    Need to specify the model based by string (because parts of the filenames are hardcoded...). 
    It can recognise (note, it is case sensistive):
        'bcc', 'ccsm4', 'csiro_mk3l', 'fgoals_gl' 'fgoals_s2', 
        'giss_21', 'giss_22', 'giss_23', 'giss_24', 'giss_25', 'giss_26', 'giss_27', 'giss_28',
        'hadcm3', 'ipsl', 'miroc', 'mpi', 'mri'

    """

    if modelname == 'bcc':
        print('...... importing bcc')
        ds = xr.open_mfdataset('%s/past1000/%s/%s_Amon_bcc-csm1-1_past1000_r1i1p1_*.nc' % (filepath_pmip3, var, var),
                               combine='by_coords', chunks={'time': 1000})
    if modelname == 'ccsm4':
        print('...... importing ccsm4')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var),
                                chunks={'time': 1000})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_CCSM4_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var),
                                chunks={'time': 1000})
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
        ds_p2['lat'] = ds_p1.lat
        ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'csiro_mk3l':
        print('...... importing csiro_mk3l')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_CSIRO-Mk3L-1-2_past1000_r1i1p1_085101-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 1000})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_CSIRO-Mk3L-1-2_historical_r1i1p1_185101-200012.nc' % (filepath_pmip3, var, var), chunks={'time': 1000})
        ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'fgoals_gl':
        print('...... importing fgoals_gl')
        ds = xr.open_dataset('%s/past1000/%s/%s_Amon_FGOALS-gl_past1000_r1i1p1_100001-199912.nc'  % (filepath_pmip3, var, var), chunks={'time': 1000})
    if modelname == 'fgoals_s2':
        print('...... importing fgoals_s2')
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
        print('...... importing giss_21')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p121_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p121_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p121_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p121_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_22':
        print('...... importing giss_22')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p122_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p122_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p122_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p122_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_23':
        print('...... importing giss_23')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p123_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p123_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p123_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p123_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_24':
        print('...... importing giss_24')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p124_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p124_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p124_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p124_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_25':
        print('...... importing giss_25')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p125_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p125_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p125_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p125_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_26':
        print('...... importing giss_26')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p126_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p126_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p126_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p126_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_27':
        print('...... importing giss_27')
        if var == 'ts' or var == 'pr':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p127_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p127_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p127_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p127_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'giss_28':
        print('...... importing giss_28')
        if var == 'ts' or var == 'pr' or var == 'psl':
            ds_p1 = xr.open_mfdataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p128_*.nc'  % (filepath_pmip3, var, var),  combine='by_coords', chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p128_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
        else:
            ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_GISS-E2-R_past1000_r1i1p128_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
            ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_GISS-E2-R_historical_r1i1p128_*.nc' % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})
            ds = xr.concat([ds_p1, ds_p2], dim='time')
    if modelname == 'hadcm3':
        print('...... importing hadcm3')
        print('...... friendly reminder that hadcm3 is missing years 1851-1859')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_HadCM3_past1000_r1i1p1_085001-185012.nc'  % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_mfdataset('%s/historical/%s/%s_Amon_HadCM3_historical_r1i1p1_*.nc'  % (filepath_pmip3, var, var), combine='by_coords', chunks={'time': 600})

        # print(ds_p1.time.dt.calendar)
        # Note: hadcm3 is in a 360_day calendar

        # --- create an empty array of NaNs for the missing years
        missing_times_begin = cftime.Datetime360Day(1851, 1, 16)
        missing_times_begin_num = cftime.date2num(missing_times_begin, units='days since 0850-01-01', calendar=ds_p1.time.dt.calendar)

        missing_times_end = cftime.Datetime360Day(1859, 11, 16)
        missing_times_end_num = cftime.date2num(missing_times_end, units='days since 0850-01-01', calendar=ds_p1.time.dt.calendar)
        missing_times = np.arange(missing_times_begin_num, missing_times_end_num + 30, 30)
        missing_times_list = cftime.num2date(missing_times, units='days since 0850-01-01', calendar=ds_p1.time.dt.calendar)
        # print(len(missing_times_list))

        # create a tmp array the same size as the missing times
        ds_tmp = ds_p1
        ds_tmp = ds_tmp.isel(time=np.arange(0,len(missing_times_list),1))

        # fill the tmp array with some NaNs
        nans = xr.zeros_like(ds_tmp.pr) * np.NaN
        ds_tmp['pr'] = xr.zeros_like(ds_tmp.pr) * np.NaN
        ds_tmp['time'] = missing_times_list

        # --- merge this all together into a single ds
        ds = xr.concat([ds_p1, ds_tmp, ds_p2], dim='time')
    if modelname == 'ipsl':
        print('...... importing ipsl')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_IPSL-CM5A-LR_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_IPSL-CM5A-LR_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600})
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeNoLeap(1851,1,1), drop=True)  # get rid of duplicate 1850 year
        
        # print(ds_p1.time.dt.calendar)
        # part 1 is in '360_day' 

        # print(ds_p2.time.dt.calendar)
        # part 2 is in 'noleap'

        # ---> IPSL uses two different calendars.. manually convert each part to monthly, before we combine to annual.
        # convert monthly to annual
        # month length is based on the calendar type of the original
        month_length_p1 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p1.pr, calendar=ds_p1.time.dt.calendar), coords=[ds_p1.pr.time], name='month_length')
        ds_p1['PRECT_mm'] = ds_p1.pr * 60 * 60 * 24 * month_length_p1 # to be in mm/month first
        ds_p1_annual = ds_p1.groupby('time.year').sum('time', skipna=False)

        month_length_p2 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p2.pr, calendar=ds_p2.time.dt.calendar), coords=[ds_p2.pr.time], name='month_length')
        ds_p2['PRECT_mm'] = ds_p2.pr * 60 * 60 * 24 * month_length_p2 # to be in mm/month first
        ds_p2_annual = ds_p2.groupby('time.year').sum('time', skipna=False)

        ###### NOTE - returning annual, not monthly data here
        ds = xr.concat([ds_p1_annual, ds_p2_annual], dim='year')
    if modelname == 'miroc':
        print('...... importing miroc')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MIROC-ESM_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MIROC-ESM_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        
        # print(ds_p1.time.dt.calendar)
        # part 1 is 'proleptic_gregorian'
        # print(ds_p2.time.dt.calendar)
        # part 2 is 'standard'

        # ---> miroc uses two different calendars.. manually convert each part to monthly, before we combine to annual.
        # convert monthly to annual
        # month length is based on the calendar type of the original
        month_length_p1 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p1.pr, calendar=ds_p1.time.dt.calendar), coords=[ds_p1.pr.time], name='month_length')
        ds_p1['PRECT_mm'] = ds_p1.pr * 60 * 60 * 24 * month_length_p1 # to be in mm/month first
        ds_p1_annual = ds_p1.groupby('time.year').sum('time', skipna=False)

        month_length_p2 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p2.pr, calendar=ds_p2.time.dt.calendar), coords=[ds_p2.pr.time], name='month_length')
        ds_p2['PRECT_mm'] = ds_p2.pr * 60 * 60 * 24 * month_length_p2 # to be in mm/month first
        ds_p2_annual = ds_p2.groupby('time.year').sum('time', skipna=False)

        ###### NOTE - returning annual, not monthly data here
        ds = xr.concat([ds_p1_annual, ds_p2_annual], dim='year')
    if modelname == 'mpi':
        print('...... importing mpi')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MPI-ESM-P_past1000_r1i1p1_085001-184912.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MPI-ESM-P_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        ds = xr.concat([ds_p1, ds_p2], dim='time')
        # print(ds_p1.time.dt.calendar)
        # 'proleptic_gregorian'
        # print(ds_p2.time.dt.calendar)
        # 'proleptic_gregorian'
    if modelname == 'mri':
        print('...... importing mri')
        ds_p1 = xr.open_dataset('%s/past1000/%s/%s_Amon_MRI-CGCM3_past1000_r1i1p1_085001-185012.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        ds_p2 = xr.open_dataset('%s/historical/%s/%s_Amon_MRI-CGCM3_historical_r1i1p1_185001-200512.nc' % (filepath_pmip3, var, var), chunks={'time': 600}, use_cftime=True)
        
        # print(ds_p1.time.dt.calendar)
        # part 1 is in 'proleptic_gregorian'
        # print(ds_p2.time.dt.calendar)
        # part 2 is in 'standard'

        # get rid of duplicate year
        ds_p2 = ds_p2.where(ds_p2.time >= cftime.DatetimeGregorian(1851,1,1), drop=True)

        # ---> mri uses two different calendars.. manually convert each part to monthly, before we combine to annual.
        # convert monthly to annual
        # month length is based on the calendar type of the original
        month_length_p1 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p1.pr, calendar=ds_p1.time.dt.calendar), coords=[ds_p1.pr.time], name='month_length')
        ds_p1['PRECT_mm'] = ds_p1.pr * 60 * 60 * 24 * month_length_p1 # to be in mm/month first
        ds_p1_annual = ds_p1.groupby('time.year').sum('time', skipna=False)      

        month_length_p2 = xr.DataArray(climate_xr_funcs.get_dpm(ds_p2.pr, calendar=ds_p2.time.dt.calendar), coords=[ds_p2.pr.time], name='month_length')
        ds_p2['PRECT_mm'] = ds_p2.pr * 60 * 60 * 24 * month_length_p2 # to be in mm/month first
        ds_p2_annual = ds_p2.groupby('time.year').sum('time', skipna=False)
        
        ###### NOTE - returning annual, not monthly data here
        ds = xr.concat([ds_p1_annual, ds_p2_annual], dim='year')
    return ds

def convert_pmip3_pr_annual(ds):
    month_length = xr.DataArray(climate_xr_funcs.get_dpm(ds.pr, calendar=ds.time.dt.calendar), coords=[ds.pr.time], name='month_length')
    ds['PRECT_mm'] = ds.pr * 60 * 60 * 24 * month_length # to be in mm/month first
    ds_annual = ds.groupby('time.year').sum('time', skipna=False)
    return ds_annual

def read_in_pmip3_and_cesmlme_control_files(modelname, var):
    """
    Read in the control files from both PMIP3 and cesmlmen and convert to annual
    """

    if modelname == 'bcc':
        bcc = xr.open_mfdataset('%s/piControl/%s/*_bcc-csm1-1_*.nc' % (filepath_pmip3, var))
        # print(bcc.time.dt.calendar)
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(bcc)

    if modelname == 'ccsm4':    
        ccsm4 = xr.open_mfdataset('%s/piControl/%s/*_CCSM4_*.nc' % (filepath_pmip3, var))
        # ccsm4.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(ccsm4)

    if modelname == 'csiro_mk3l':
        csiro_mk3l = xr.open_mfdataset('%s/piControl/%s/*_CSIRO-Mk3L-1-2_*.nc' % (filepath_pmip3, var))
        # csiro_mk3l.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(csiro_mk3l)

    # if modelname == 'fgoals_gl'
    # I can't find this? 
    if modelname == 'fgoals_s2':
        fgoals_s2 = xr.open_mfdataset('%s/piControl/%s/*_FGOALS-s2_*.nc' % (filepath_pmip3, var))
        # fgoals_s2.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(fgoals_s2)

    if modelname == 'giss_1':
        giss_1 = xr.open_mfdataset('%s/piControl/%s/*_GISS-E2-R_piControl_r1i1p1_*.nc' % (filepath_pmip3, var))
        # giss_2.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(giss_1)

    if modelname == 'giss_2':
        giss_2 = xr.open_mfdataset('%s/piControl/%s/*_GISS-E2-R_piControl_r1i1p2_*.nc' % (filepath_pmip3, var))
        # giss_2.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(giss_2)

    if modelname == 'giss_3':
        giss_3 = xr.open_mfdataset('%s/piControl/%s/*_GISS-E2-R_piControl_r1i1p3_*.nc' % (filepath_pmip3, var))
        # giss_3.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(giss_3)

    if modelname == 'giss_41':
        giss_41 = xr.open_mfdataset('%s/piControl/%s/*_GISS-E2-R_piControl_r1i1p141_*.nc' % (filepath_pmip3, var))
        ds_annual = convert_pmip3_pr_annual(giss_41)

    if modelname == 'hadcm3':
        hadcm3 = xr.open_mfdataset('%s/piControl/%s/*_HadCM3_*.nc' % (filepath_pmip3, var))
        # hadcm3.time.dt.calendar
        # '360_day'
        ds_annual = convert_pmip3_pr_annual(hadcm3)

    if modelname == 'ipsl':
        ipsl = xr.open_mfdataset('%s/piControl/%s/*_IPSL-CM5A-LR_*.nc' % (filepath_pmip3, var))
        # ipsl.time.dt.calendar
        # 'noleap'
        ds_annual = convert_pmip3_pr_annual(ipsl)

    if modelname == 'miroc':
        miroc = xr.open_mfdataset('%s/piControl/%s/*_MIROC-ESM_*.nc' % (filepath_pmip3, var))
        # miroc.time.dt.calendar
        # 'standard'
        ds_annual = convert_pmip3_pr_annual(miroc)

    if modelname == 'mpi':
        mpi = xr.open_mfdataset('%s/piControl/%s/*_MPI-ESM-P_*.nc' % (filepath_pmip3, var), use_cftime=True)
        # mpi.time.dt.calendar
        # 'proleptic_gregorian'
        ds_annual = convert_pmip3_pr_annual(mpi)

    if modelname == 'mri':
        mri = xr.open_mfdataset('%s/piControl/%s/*_MRI-CGCM3_*.nc' % (filepath_pmip3, var))
        # mri.time.dt.calendar
        # 'standard'
        ds_annual = convert_pmip3_pr_annual(mri)

    if modelname == 'cesmlme':
        cesmlme = climate_xr_funcs.import_control_variable_cam(filepath, 'PRECT')
        ds_annual = cesmlme.groupby('time.year').sum('time', skipna=False)

    return ds_annual

# ---- Functions to apply drought metrics
def droughts_historical_fromyear(ds, historical_year, output_dir, model_filename):

    """ Calculate drought metrics for the historical models

    This function will drop any data above the year 2000 (for convenience, because some of the PMIP3 stopped at 2000 and some at 2005),
    and calculate droughts across the historical year--<2000 period, relative to the entire historical period as a climatology. 

    There are a few different ways it calculates droughts.
    Preferred here is '2S2E', but it can also calculate any years below a threshold (e.g. median, 20% climatology precip), 
    OR a drought starts after 1 year below a threshold (20%), and ends 2 years above a threshold (median), 
    OR a drought starts 2 years below a threshold (20%) and ends one year above a threshold (median). Was just playing around with options..
    The definition will have an impact on how long the other metrics are.

    Inputs:
        - ds: as an xarray dataset, with PRECT_mm as a variable for your annual precipitation data
        - historical_year: The start year of your historical period you want to use. e.g. 1850 or 1900.
          We only had observations for 1900 onwards and wanted to compare the output here with obs,
          which is why we subsetted our historical period to 1900 onwards.
        - output_dir: the output directory (as string) to save your netcdf file with drought metrics for X model.
          Folder will get created by this function.
        - model_filename: model name (as string), used for saving the netcdf files

    Outputs:
        - netcdf files per model with drought metrics for last millennium
        - drought metrics include:
            - mean length of drought
            - max length of drought
            - intensity of droughts
            - severity
        - droughts are calculated using a variety of thresholds. These can be adjusted if you like, use at your own risk.

    """

    print('...... Calculating drought metrics for %s from %s onwards' % (model_filename, historical_year))
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
    ds_hist = ds_hist.drop_vars('quantile')
    
    # --- Create the output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # --- Save netcdf file. Use some compresion so things don't take lots of space
    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_hist.data_vars}
    
    print('...... Saving %s historical file to:  %s' % (model_filename, output_dir))
    
    delayed_obj = ds_hist.to_netcdf('%s/%s_precip_hist_annual.nc' % (output_dir, model_filename), compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()
    
    return ds_hist

def droughts_lm_thresholdyears(ds, threshold_startyear, threshold_endyear, output_dir, model_filename):
    """ Calculate drought metrics for the last millennium models

    This function will drop any data above the year 2000 (for convenience, because some of the PMIP3 stopped at 2000 and some at 2005),
    and calculate droughts across the 850-2000 period, relative to a climatology specified by the user (by threshold_startyear, threshold_endyear)
    
    There are a few different ways it calculates droughts.
    Preferred here is '2S2E', but it can also calculate any years below a threshold (e.g. median, 20% climatology precip), 
    OR a drought starts after 1 year below a threshold (20%), and ends 2 years above a threshold (median), 
    OR a drought starts 2 years below a threshold (20%) and ends one year above a threshold (median). Was just playing around with options..
    The definition will have an impact on how long the other metrics are.

    Inputs:
        - ds: as an xarray dataset, with PRECT_mm as a variable for your annual precipitation data
        - lm_threshold_startyear: The start year you want to calculate your last millennium droughts relative to
          (e.g. 1900, or 850 if using entire lm period). Basically start of your climatology period. 
        - lm_threshold_endyear: The end year you want to calculate your last millennium droughts relative to
          (e.g. 2000, or perhaps 1850 if using lm period). Basically end of your climatology period.
        - output_dir: the output directory (as string) to save your netcdf file with drought metrics for X model.
          Folder will get created by this function.
        - model_filename: model name (as string), used for saving the netcdf files

    Outputs:
        - netcdf files per model with drought metrics for last millennium
        - drought metrics include:
            - mean length of drought
            - max length of drought
            - intensity of droughts
            - severity
        - droughts are calculated using a variety of thresholds. These can be adjusted if you like, use at your own risk.
    
    """
    print('...... Calculating drought metrics for %s using %s-%s as climatology' % (model_filename, threshold_startyear, threshold_endyear))
    
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
    
    # intensity  - relative to climatological mean (same as in Anna's paper)
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
    ds_lm = ds_lm.drop_vars('quantile')
    
    # --- Create the output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # --- Save netcdf file. Use some compresion so things don't take lots of space
    print('...... Saving %s last millennium file to:  %s' % (model_filename, output_dir))
    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds_lm.data_vars}
    
    delayed_obj = ds_lm.to_netcdf('%s/%s_precip_lm_annual.nc' % (output_dir, model_filename), compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()
    
    return ds_lm

def drought_metrics_control(ds, output_dir, model_filename):
    """ Calculate drought metrics for the control models
    (or anything really - this version will calculate to the long-term mean of the input dataset)

    There are a few different ways it calculates droughts.
    Preferred here is '2S2E', but it can also calculate any years below a threshold (e.g. median, 20% climatology precip), 
    OR a drought starts after 1 year below a threshold (20%), and ends 2 years above a threshold (median), 
    OR a drought starts 2 years below a threshold (20%) and ends one year above a threshold (median). Was just playing around with options..
    The definition will have an impact on how long the other metrics are.

    Inputs:
        - ds: as an xarray dataset, with PRECT_mm as a variable for your annual precipitation data
        - output_dir: the output directory (as string) to save your netcdf file with drought metrics for X model.
          Folder will get created by this function.
        - model_filename: model name (as string), used for saving the netcdf files

    Outputs:
        - netcdf files per model with drought metrics for last millennium
        - drought metrics include:
            - mean length of drought
            - max length of drought
            - intensity of droughts
            - severity
        - droughts are calculated using a variety of thresholds. These can be adjusted if you like, use at your own risk.
    
    """

    print('...... Calculating drought metrics for %s' % (model_filename))
    # stats are calculated relative to the long term mean
    
    # get years that are drought vs not drought
    ds['drought_years_2s2e']       = climate_droughts_xr_funcs.get_drought_years_2S2E_apply(ds.PRECT_mm, ds.PRECT_mm.mean(dim='year'))
    ds['drought_years_median']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds.PRECT_mm, ds.PRECT_mm.quantile(0.5, dim=('year')))
    ds['drought_years_20perc']     = climate_droughts_xr_funcs.get_drought_years_below_threshold_apply(ds.PRECT_mm, ds.PRECT_mm.quantile(0.2, dim=('year')))
    ds['drought_years_120pc_2med'] = climate_droughts_xr_funcs.get_drought_years_120perc_2median_apply(ds.PRECT_mm, ds.PRECT_mm.quantile(0.2, dim='year'), ds.PRECT_mm.quantile(0.5, dim='year'))
    ds['drought_years_220pc_1med'] = climate_droughts_xr_funcs.get_drought_years_start_end_thresholds_apply(ds.PRECT_mm, ds.PRECT_mm.quantile(0.2, dim=('year')), ds.PRECT_mm.quantile(0.5, dim=('year')))
    
    # print('...... Calculated drought years')
    # get overall length of droughts
    ds['droughts_2s2e']       = climate_droughts_xr_funcs.cumulative_drought_length(ds['drought_years_2s2e'])
    ds['droughts_median']     = climate_droughts_xr_funcs.cumulative_drought_length(ds['drought_years_median'])
    ds['droughts_20perc']     = climate_droughts_xr_funcs.cumulative_drought_length(ds['drought_years_20perc'])
    ds['droughts_120pc_2med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds['drought_years_120pc_2med'])
    ds['droughts_220pc_1med'] = climate_droughts_xr_funcs.cumulative_drought_length(ds['drought_years_220pc_1med'])
    
    # print('...... Calculated the max length of droughts')
    # get max length in this period
    ds['droughts_2s2e_max']       = climate_droughts_xr_funcs.max_length_ufunc(ds.droughts_2s2e, dim='year')
    ds['droughts_median_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds.droughts_median, dim='year')
    ds['droughts_20perc_max']     = climate_droughts_xr_funcs.max_length_ufunc(ds.droughts_20perc, dim='year')
    ds['droughts_120pc_2med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds.droughts_120pc_2med, dim='year')
    ds['droughts_220pc_1med_max'] = climate_droughts_xr_funcs.max_length_ufunc(ds.droughts_220pc_1med, dim='year')

    # print('...... Calculated the max length of drought')
    # get mean length in this period
    ds['droughts_2s2e_mean']       = climate_droughts_xr_funcs.mean_length_ufunc(ds.droughts_2s2e, dim='year')
    ds['droughts_median_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds.droughts_median, dim='year')
    ds['droughts_20perc_mean']     = climate_droughts_xr_funcs.mean_length_ufunc(ds.droughts_20perc, dim='year')
    ds['droughts_120pc_2med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds.droughts_120pc_2med, dim='year')
    ds['droughts_220pc_1med_mean'] = climate_droughts_xr_funcs.mean_length_ufunc(ds.droughts_220pc_1med, dim='year')

    # print('...... Calculated the mean length of drought')
    # count how many individual events occur
    ds['droughts_2s2e_no_of_events']       = climate_droughts_xr_funcs.count_drought_events_apply(ds.droughts_2s2e)
    ds['droughts_median_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds.droughts_median)
    ds['droughts_20perc_no_of_events']     = climate_droughts_xr_funcs.count_drought_events_apply(ds.droughts_20perc)
    ds['droughts_120pc_2med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds.droughts_120pc_2med)
    ds['droughts_220pc_1med_no_of_events'] = climate_droughts_xr_funcs.count_drought_events_apply(ds.droughts_220pc_1med)
    
    # print('...... Counted number of drought events')
    # std
    ds['droughts_2s2e_std']       = climate_droughts_xr_funcs.std_apply(ds.droughts_2s2e, dim='year')
    ds['droughts_median_std']     = climate_droughts_xr_funcs.std_apply(ds.droughts_median, dim='year')
    ds['droughts_20perc_std']     = climate_droughts_xr_funcs.std_apply(ds.droughts_20perc, dim='year')
    ds['droughts_120pc_2med_std'] = climate_droughts_xr_funcs.std_apply(ds.droughts_120pc_2med, dim='year')
    ds['droughts_220pc_1med_std'] = climate_droughts_xr_funcs.std_apply(ds.droughts_220pc_1med, dim='year')
    
    # print('...... Calculated drought std')   
    # intensity  - relative to climatological mean (same as in anna's paper)
    ds['droughts_2s2e_intensity']        = climate_droughts_xr_funcs.drought_intensity(ds, 'drought_years_2s2e', 'droughts_2s2e', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_median_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds, 'drought_years_median', 'droughts_median', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_20perc_intensity']      = climate_droughts_xr_funcs.drought_intensity(ds, 'drought_years_20perc', 'droughts_20perc', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_120pc_2med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_220pc_1med_intensity'] = climate_droughts_xr_funcs.drought_intensity(ds, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds.PRECT_mm.mean(dim='year'))
    
    # print('...... Calculated drought intensity')
    # severity - intensity x length
    ds['droughts_2s2e_severity']        = climate_droughts_xr_funcs.drought_severity(ds, 'drought_years_2s2e', 'droughts_2s2e', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_median_severity']      = climate_droughts_xr_funcs.drought_severity(ds, 'drought_years_median', 'droughts_median', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_20perc_severity']      = climate_droughts_xr_funcs.drought_severity(ds, 'drought_years_20perc', 'droughts_20perc', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_120pc_2med_severity'] = climate_droughts_xr_funcs.drought_severity(ds, 'drought_years_120pc_2med', 'droughts_120pc_2med', ds.PRECT_mm.mean(dim='year'))
    ds['droughts_220pc_1med_severity'] = climate_droughts_xr_funcs.drought_severity(ds, 'drought_years_220pc_1med', 'droughts_220pc_1med', ds.PRECT_mm.mean(dim='year'))
    
    # print('...... Calculated drought severity')
    ds['droughts_2s2e_no_events_100yrs'] = (ds.droughts_2s2e_no_of_events / len(ds.year)) * 100
    ds['droughts_2s2e_sum'] = climate_droughts_xr_funcs.sum_apply(ds.droughts_2s2e, dim='year') 
    ds['droughts_2s2e_sum_100yrs'] = climate_droughts_xr_funcs.sum_apply(ds.droughts_2s2e, dim='year') / len(ds.year) * 100
    
    # get rid of quantile
    ds = ds.drop_vars('quantile')
    
    # --- Create the output directory
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    
    # --- Save netcdf file. Use some compresion so things don't take lots of space    
    print('... Saving %s file to:  %s' % (model_filename, output_dir))
    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds.data_vars}
    
    delayed_obj = ds.to_netcdf('%s/%s_precip_cntl_annual.nc' % (output_dir, model_filename), compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()
    
    return ds

def process_pmip3_files(ds, modelname, historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir):
    """ Calculate drought metrics for all the PMIP3 models for the historical and last millennium periods.
    
    Inputs:
        - ds: your input monthly dataset, with 'pr' as a variable (this will get converted to 'PRECT_mm' (per year) 
              NOTE: for a few models, I use the annual dataset with PRECT_mm already calculated, because their calendars were super annoying.
        - modelname: string of the model name, used for saving the netcdf files
        
        - historical_year: what year is the START of your historical period, e.g., 1850 or 1900, etc. 
          This is for subsetting your dataset for the historical period, and gets passed to the function `droughts_historical_fromyear`
        - hist_output_dir: the output directory to save your netcdf file with drought metrics for X model. 
          Folder will get created by `droughts_historical_fromyear`
        
        - lm_threshold_startyear: The start year you want to calculate your last millennium droughts relative to (e.g. 1900, or 850 if using entire lm period).
          Basically start of your climatology period. Passed to function `droughts_lm_thresholdyears`
        - lm_threshold_endyear: The end year you want to calculate your last millennium droughts relative to (e.g. 2000, or perhaps 1850 if using lm period).
          Basically end of your climatology period. Passed to function `droughts_lm_thresholdyears`
        - lm_output_dir: the output directory to save your netcdf file with drought metrics for X model.
          Folder will get created by `droughts_lm_thresholdyears`
    
    Outputs:
        - netcdf files per model with drought metrics for:
            - historical period
            - last millennium
        - drought metrics include:
            - mean length of drought
            - max length of drought
            - intensity of droughts
            - severity
        - droughts are calculated using a variety of thresholds. These can be adjusted.

    """

    # --- Skip models that are already converted in annual datasets.
    if modelname == 'ipsl':
        # ds is already in annual, because calendar was annoying.
        ds_annual = ds
        ds_annual.load()
    elif modelname == 'miroc':
        # ds is already in annual, because calendar was annoying.
        ds_annual = ds
        ds_annual.load()
    elif modelname == 'mri':
        # ds is already in annual, because calendar was annoying.
        ds_annual = ds
        ds_annual.load()
    else:
        # import MONTHLY pmip3 and convert to annual
        month_length = xr.DataArray(climate_xr_funcs.get_dpm(ds.pr, calendar=ds.time.dt.calendar), coords=[ds.pr.time], name='month_length')
        ds['PRECT_mm'] = ds.pr * 60 * 60 * 24 * month_length # to be in mm/month first
        ds_annual = ds.groupby('time.year').sum('time', skipna=False)
        ds_annual.load()
    

    # --- run the drought metric workflow based on ANNUAL data
    # process for historical
    ds_hist = droughts_historical_fromyear(ds_annual, historical_year, hist_output_dir, modelname)
    
    # process for last millennium droughts
    ds_lm = droughts_lm_thresholdyears(ds_annual, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir, modelname)
    
    return ds_hist, ds_lm

def process_cesm_files(ds, modelname, historical_year, hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir):
    """ Calculate drought metrics for CESM-LME models for the historical and last millennium periods.
    
    Inputs:
        - ds: your input ANNUAL dataset, with 'PRECT_mm' as a variable. When importing CESM (with `import_cesmlme`), it was already converted into annual
        - modelname: string of the model name (or ensemble number), used for saving the netcdf files
        
        - historical_year: what year is the START of your historical period, e.g., 1850 or 1900, etc. 
          This is for subsetting your dataset for the historical period, and gets passed to the function `droughts_historical_fromyear`
        - hist_output_dir: the output directory to save your netcdf file with drought metrics for X model. 
          Folder will get created by `droughts_historical_fromyear`
        
        - lm_threshold_startyear: The start year you want to calculate your last millennium droughts relative to (e.g. 1900, or 850 if using entire lm period).
          Basically start of your climatology period. Passed to function `droughts_lm_thresholdyears`
        - lm_threshold_endyear: The end year you want to calculate your last millennium droughts relative to (e.g. 2000, or perhaps 1850 if using lm period).
          Basically end of your climatology period. Passed to function `droughts_lm_thresholdyears`
        - lm_output_dir: the output directory to save your netcdf file with drought metrics for X model.
          Folder will get created by `droughts_lm_thresholdyears`
    
    Outputs:
        - netcdf files per model with drought metrics for:
            - historical period
            - last millennium
        - drought metrics include:
            - mean length of drought
            - max length of drought
            - intensity of droughts
            - severity
        - droughts are calculated using a variety of thresholds. These can be adjusted.

    """

    # --- run the drought metric workflow based on ANNUAL data
    ds_hist = droughts_historical_fromyear(ds, historical_year, hist_output_dir, modelname)

    # process for last millennium
    ds_lm = droughts_lm_thresholdyears(ds, lm_threshold_startyear, lm_threshold_endyear, lm_output_dir, modelname)
    
    return ds_hist, ds_lm	

# Other things that are very helpful here
def get_aus(ds):
    mask = regionmask.defined_regions.natural_earth.countries_110.mask(ds)
    ds_aus = ds.where(mask == 137, drop=True)
    return ds_aus

def save_netcdf_compression(ds, output_dir, filename):
    # Function to save netcdf files with some compression, so files don't take up a ridiculous amount of space
    # specify the output directory and filename when saving, it will add the .nc when saving.
    # can turn down the complevel to 5 or 6 if needed.

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds.data_vars}

    delayed_obj = ds.to_netcdf('%s/%s.nc' % (output_dir, filename), mode='w', compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()


def regrid_files(ds):
    """
    Regrid the files to a different grid spacing and bilinear interpolation using xesmf.
    Here it is hardcoded using 2° x 2°.

    # Instructions for regridding using xesmf are here: https://xesmf.readthedocs.io/en/latest/notebooks/Dataset.html
    
    Input: xarray dataset
    Output: xarray dataset
    """

    # hardcoded to be 2° output
    ds_out = xr.Dataset({'lat': (['lat'], np.arange(-90, 92, 2)),
                         'lon': (['lon'], np.arange(0, 360, 2))})
    
    regridder = xe.Regridder(ds, ds_out, 'bilinear', periodic=False)
   
    ds_out = regridder(ds)
    #     for k in ds.data_vars:
    #         print(k, ds_out[k].equals(regridder(ds[k])))
    return ds_out

def calculate_drought_sum(ds):
    # calculate drought sum - I must have forgotten to do this earlier.
    ds['droughts_2s2e_sum_100yrs'] = (ds.droughts_2s2e.sum(dim='year') / len(ds.year)) * 100
    ds['droughts_2s2e_no_events_100yrs'] = (ds.droughts_2s2e_no_of_events / len(ds.year)) * 100
    return ds
# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Check if some of the output directories exists, otherwise make them.

if not os.path.exists(hist_output_dir):
    print("... Creating %s now "  % hist_output_dir)
    os.makedirs(hist_output_dir)

if not os.path.exists(lm_output_dir):
    print("... Creating %s now "  % lm_output_dir)
    os.makedirs(lm_output_dir)

if not os.path.exists(cntl_output_dir):
    print("... Creating %s now "  % cntl_output_dir)
    os.makedirs(cntl_output_dir)


# create subfolders to try and organise things a little...
if not os.path.exists('%s/global' % hist_output_dir):
    os.makedirs('%s/global' % hist_output_dir)
if not os.path.exists('%s/aus' % hist_output_dir):
    os.makedirs('%s/aus' % hist_output_dir)
# if not os.path.exists('%s/sig_tests' % hist_output_dir):
#     os.makedirs('%s/sig_tests' % hist_output_dir)

if not os.path.exists('%s/global' % lm_output_dir):
    os.makedirs('%s/global' % lm_output_dir)
if not os.path.exists('%s/aus' % lm_output_dir):
    os.makedirs('%s/aus' % lm_output_dir)
# if not os.path.exists('%s/sig_tests' % lm_output_dir):
#     os.makedirs('%s/sig_tests' % lm_output_dir)


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Process CESM-LME files

if process_cesm_fullforcing_files is True:
    print('Processing CESM-LME full forcing files')
    # # # ---------------------------------
    print('... importing CESM-LME files')
    ff1_precip_annual = import_cesmlme(filepath, '001')
    ff1_precip_hist_annual , ff1_precip_lm_annual  = process_cesm_files(ff1_precip_annual, 'cesmlme-ff1', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff2_precip_annual = import_cesmlme(filepath, '002')
    ff2_precip_hist_annual , ff2_precip_lm_annual  = process_cesm_files(ff2_precip_annual, 'cesmlme-ff2', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff3_precip_annual = import_cesmlme(filepath, '003')
    ff3_precip_hist_annual , ff3_precip_lm_annual  = process_cesm_files(ff3_precip_annual, 'cesmlme-ff3', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff4_precip_annual = import_cesmlme(filepath, '004')
    ff4_precip_hist_annual , ff4_precip_lm_annual  = process_cesm_files(ff4_precip_annual, 'cesmlme-ff4', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff5_precip_annual = import_cesmlme(filepath, '005')
    ff5_precip_hist_annual , ff5_precip_lm_annual  = process_cesm_files(ff5_precip_annual, 'cesmlme-ff5', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff6_precip_annual = import_cesmlme(filepath, '006')
    ff6_precip_hist_annual , ff6_precip_lm_annual  = process_cesm_files(ff6_precip_annual, 'cesmlme-ff6', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff7_precip_annual = import_cesmlme(filepath, '007')
    ff7_precip_hist_annual , ff7_precip_lm_annual  = process_cesm_files(ff7_precip_annual, 'cesmlme-ff7', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff8_precip_annual = import_cesmlme(filepath, '008')
    ff8_precip_hist_annual , ff8_precip_lm_annual  = process_cesm_files(ff8_precip_annual, 'cesmlme-ff8', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff9_precip_annual = import_cesmlme(filepath, '009')
    ff9_precip_hist_annual , ff9_precip_lm_annual  = process_cesm_files(ff9_precip_annual, 'cesmlme-ff9', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff10_precip_annual = import_cesmlme(filepath, '010')
    ff10_precip_hist_annual, ff10_precip_lm_annual = process_cesm_files(ff10_precip_annual, 'cesmlme-ff10', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff11_precip_annual = import_cesmlme(filepath, '011')
    ff11_precip_hist_annual, ff11_precip_lm_annual = process_cesm_files(ff11_precip_annual, 'cesmlme-ff11', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff12_precip_annual = import_cesmlme(filepath, '012')
    ff12_precip_hist_annual, ff12_precip_lm_annual = process_cesm_files(ff12_precip_annual, 'cesmlme-ff12', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    ff13_precip_annual = import_cesmlme(filepath, '013')
    ff13_precip_hist_annual, ff13_precip_lm_annual = process_cesm_files(ff13_precip_annual, 'cesmlme-ff13', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    print('... Finished processing CESM-LME full forcing files!')
else:
    print('... Skipping initial processing of CESM-LME full forcing files')

# actually process the single forcing CESM files
if process_cesm_singleforcing_files is True:
    print('Processing CESM-LME single forcing files')
    # ---------------------------------
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

    # volcanoes! boom!
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
# Import PMIP3 files

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
    ccsm4_pr['time'] = giss_21_pr.time  # is this needed???

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # # # Process PMIP3 files
    # # # ---------------------------------
    # # # Process for the historical period
    bcc_precip_hist_annual       , bcc_precip_lm_annual        = process_pmip3_files(bcc_pr, 'bcc', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    ccsm4_precip_hist_annual     , ccsm4_precip_lm_annual      = process_pmip3_files(ccsm4_pr, 'ccsm4', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    csiro_mk3l_precip_hist_annual, csiro_mk3l_precip_lm_annual = process_pmip3_files(csiro_mk3l_pr, 'csiro_mk3l',historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    fgoals_gl_precip_hist_annual , fgoals_gl_precip_lm_annual  = process_pmip3_files(fgoals_gl_pr, 'fgoals_gl', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    fgoals_s2_precip_hist_annual , fgoals_s2_precip_lm_annual  = process_pmip3_files(fgoals_s2_pr, 'fgoals_s2', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_21_precip_hist_annual   , giss_21_precip_lm_annual    = process_pmip3_files(giss_21_pr, 'giss_21', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_22_precip_hist_annual   , giss_22_precip_lm_annual    = process_pmip3_files(giss_22_pr, 'giss_22', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_23_precip_hist_annual   , giss_23_precip_lm_annual    = process_pmip3_files(giss_23_pr, 'giss_23', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_24_precip_hist_annual   , giss_24_precip_lm_annual    = process_pmip3_files(giss_24_pr, 'giss_24', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_25_precip_hist_annual   , giss_25_precip_lm_annual    = process_pmip3_files(giss_25_pr, 'giss_25', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_26_precip_hist_annual   , giss_26_precip_lm_annual    = process_pmip3_files(giss_26_pr, 'giss_26', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_27_precip_hist_annual   , giss_27_precip_lm_annual    = process_pmip3_files(giss_27_pr, 'giss_27', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    giss_28_precip_hist_annual   , giss_28_precip_lm_annual    = process_pmip3_files(giss_28_pr, 'giss_28', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    hadcm3_precip_hist_annual    , hadcm3_precip_lm_annual     = process_pmip3_files(hadcm3_pr, 'hadcm3', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    ipsl_precip_hist_annual      , ipsl_precip_lm_annual       = process_pmip3_files(ipsl_pr, 'ipsl', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    miroc_precip_hist_annual     , miroc_precip_lm_annual      = process_pmip3_files(miroc_pr, 'miroc', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    mpi_precip_hist_annual       , mpi_precip_lm_annual        = process_pmip3_files(mpi_pr, 'mpi', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)
    mri_precip_hist_annual       , mri_precip_lm_annual        = process_pmip3_files(mri_pr, 'mri', historical_year, '%s/global' % hist_output_dir, lm_threshold_startyear, lm_threshold_endyear, '%s/global' % lm_output_dir)

    print('... Finished processing PMIP3 files!')
else:
    print('... Skipping initial processing of PMIP3 files')


# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------ Calculate ensemble means
if calculate_cesm_ff_ens_means is True:
    print('... Calculating CESM-LME full forcing ensemble mean')
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
    print('... Calculating CESM-LME single forcing ensemble means')
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
    print('... Calculating GISS ensemble mean')
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
    print('... subsetting PMIP3 historical files to Aus only')
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
    # giss_all_precip_hist_annual   = xr.open_dataset('%s/global/giss_all_precip_hist_annual.nc' % hist_output_dir)
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
    # giss_all_precip_hist_annual_aus = get_aus(giss_all_precip_hist_annual)
    # save_netcdf_compression(giss_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'giss_all_precip_hist_annual_aus')
else:
    pass

if subset_lme_ff_hist_files_to_Aus_only is True:
    print('... subsetting CESM-LME full-forcing historical files to Aus only')
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
    # ff_all_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ff_all_precip_hist_annual.nc' % hist_output_dir)

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
    
    # # ensemble mean
    # ff_all_precip_hist_annual_aus = get_aus(ff_all_precip_hist_annual)
    # save_netcdf_compression(ff_all_precip_hist_annual_aus, hist_output_dir + '/aus', 'cesmlme-ff_all_precip_hist_annual_aus')
else:
    pass


if subset_lme_single_forcing_hist_files_to_Aus_only is True:
    print('... subsetting CESM-LME single forcing historical files to Aus only')
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
    print('... subsetting PMIP3 last millennium files to Aus only')
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
    # giss_all_precip_lm_annual_aus = xr.open_dataset('%s/global/giss_all_precip_lm_annual.nc' % lm_output_dir)
    
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
    # giss_all_precip_lm_annual_aus = get_aus(giss_all_precip_lm_annual)
    # save_netcdf_compression(giss_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'giss_all_precip_lm_annual_aus')
else:
    pass

if subset_lme_ff_lm_files_to_Aus_only is True:
    print('... subsetting CESM-LME full-forcing last millennium files to Aus only')
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
    # ff_all_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ff_all_precip_lm_annual.nc' % lm_output_dir)

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
    
    # ff_all_precip_lm_annual_aus = get_aus(ff_all_precip_lm_annual)
    # save_netcdf_compression(ff_all_precip_lm_annual_aus, lm_output_dir + '/aus', 'cesmlme-ff_all_precip_lm_annual_aus')
else:
    pass

if subset_lme_single_forcing_lm_files_to_Aus_only is True:
    print('... subsetting CESM-LME single forcing last millennium files to Aus only')
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


# ---------------------
# ----- AWAP
if process_awap is True:
    print('... Processing AWAP')
    # AWAP is from Anna
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

    # awap_gf_annual = awap_gf_annual.where(awap_gf_annual['year'] <= 2000, drop=True)

    awap_gf_annual = droughts_historical_fromyear(awap_gf_annual, historical_year, hist_output_dir, 'awap_gf')
    save_netcdf_compression(awap_gf_annual, hist_output_dir, 'awap_gf_annual')

else:
    print('... Skipping processing of AWAP')
    pass

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

# Control files
if process_control_files is True:
    print('... Processing control files')
    # read in control files and convert to annual
    bcc_cntl = read_in_pmip3_and_cesmlme_control_files('bcc', 'pr')
    bcc_cntl.load()
    bcc_annual_cntl  = drought_metrics_control(bcc_cntl,  cntl_output_dir, 'bcc')

    ccsm4_cntl  = read_in_pmip3_and_cesmlme_control_files('ccsm4', 'pr')
    ccsm4_cntl.load()
    ccsm4_annual_cntl  = drought_metrics_control(ccsm4_cntl, cntl_output_dir, 'ccsm4')
    
    csiro_mk3l_cntl  = read_in_pmip3_and_cesmlme_control_files('csiro_mk3l', 'pr')
    csiro_mk3l_cntl.load()
    csiro_mk3l_annual_cntl = drought_metrics_control(csiro_mk3l_cntl, cntl_output_dir, 'csiro_mk3l')

    fgoals_s2_cntl  = read_in_pmip3_and_cesmlme_control_files('fgoals_s2', 'pr')
    fgoals_s2_cntl.load()
    fgoals_s2_annual_cntl = drought_metrics_control(fgoals_s2_cntl, cntl_output_dir, 'fgoals_s2')

    giss_1_cntl  = read_in_pmip3_and_cesmlme_control_files('giss_1', 'pr')
    giss_1_cntl.load()
    giss_1_annual_cntl = drought_metrics_control(giss_1_cntl, cntl_output_dir, 'giss_1')

    giss_2_cntl  = read_in_pmip3_and_cesmlme_control_files('giss_2', 'pr')
    giss_2_cntl.load()
    giss_2_annual_cntl = drought_metrics_control(giss_2_cntl, cntl_output_dir, 'giss_2')

    giss_3_cntl  = read_in_pmip3_and_cesmlme_control_files('giss_3', 'pr')
    giss_3_cntl.load()
    giss_3_annual_cntl = drought_metrics_control(giss_3_cntl, cntl_output_dir, 'giss_3')

    giss_41_cntl  = read_in_pmip3_and_cesmlme_control_files('giss_41', 'pr')
    giss_41_cntl.load()
    giss_41_annual_cntl = drought_metrics_control(giss_41_cntl, cntl_output_dir, 'giss_41')

    hadcm3_cntl  = read_in_pmip3_and_cesmlme_control_files('hadcm3', 'pr')
    hadcm3_cntl.load()
    hadcm3_annual_cntl = drought_metrics_control(hadcm3_cntl, cntl_output_dir, 'hadcm3')

    ipsl_cntl  = read_in_pmip3_and_cesmlme_control_files('ipsl', 'pr')
    ipsl_cntl.load()
    ipsl_annual_cntl = drought_metrics_control(ipsl_cntl, cntl_output_dir, 'ipsl')

    miroc_cntl  = read_in_pmip3_and_cesmlme_control_files('miroc', 'pr')
    miroc_cntl.load()
    miroc_annual_cntl = drought_metrics_control(miroc_cntl, cntl_output_dir, 'miroc')

    mpi_cntl  = read_in_pmip3_and_cesmlme_control_files('mpi', 'pr')
    mpi_cntl.load()
    mpi_annual_cntl = drought_metrics_control(mpi_cntl, cntl_output_dir, 'mpi')

    mri_cntl  = read_in_pmip3_and_cesmlme_control_files('mri', 'pr')
    mri_cntl.load()
    mri_annual_cntl = drought_metrics_control(mri_cntl, cntl_output_dir, 'mri')

    cesmlme_cntl = read_in_pmip3_and_cesmlme_control_files('cesmlme', 'PRECT')
    cesmlme_cntl.load()
    cesmlme_annual_cntl = drought_metrics_control(cesmlme_cntl, cntl_output_dir, 'cesmlme')

else:
    print('... Skipping initial processing of control files')
    pass

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ 

if subset_control_files_to_Aus is True:
    print('... Subset control files to Aus only')
    # -----------
    # read in processed files
    bcc_annual_cntl  = xr.open_dataset('%s/bcc_precip_cntl_annual.nc' % cntl_output_dir)
    ccsm4_annual_cntl = xr.open_dataset('%s/ccsm4_precip_cntl_annual.nc' % cntl_output_dir)
    csiro_mk3l_annual_cntl = xr.open_dataset('%s/csiro_mk3l_precip_cntl_annual.nc' % cntl_output_dir)
    fgoals_s2_annual_cntl  = xr.open_dataset('%s/fgoals_s2_precip_cntl_annual.nc' % cntl_output_dir)
    giss_1_annual_cntl = xr.open_dataset('%s/giss_1_precip_cntl_annual.nc' % cntl_output_dir)
    giss_2_annual_cntl = xr.open_dataset('%s/giss_2_precip_cntl_annual.nc' % cntl_output_dir)
    giss_3_annual_cntl = xr.open_dataset('%s/giss_3_precip_cntl_annual.nc' % cntl_output_dir)
    giss_41_annual_cntl = xr.open_dataset('%s/giss_41_precip_cntl_annual.nc' % cntl_output_dir)
    hadcm3_annual_cntl = xr.open_dataset('%s/hadcm3_precip_cntl_annual.nc' % cntl_output_dir)
    ipsl_annual_cntl  = xr.open_dataset('%s/ipsl_precip_cntl_annual.nc' % cntl_output_dir)
    miroc_annual_cntl = xr.open_dataset('%s/miroc_precip_cntl_annual.nc' % cntl_output_dir)
    mpi_annual_cntl  = xr.open_dataset('%s/mpi_precip_cntl_annual.nc' % cntl_output_dir)
    mri_annual_cntl  = xr.open_dataset('%s/mri_precip_cntl_annual.nc' % cntl_output_dir)
    cesmlme_annual_cntl   = xr.open_dataset('%s/cesmlme_precip_cntl_annual.nc' % cntl_output_dir)

    # actually Subset the files
    bcc_annual_cntl_aus = get_aus(bcc_annual_cntl)
    save_netcdf_compression(bcc_annual_cntl_aus, cntl_output_dir, 'bcc_precip_cntl_annual_aus')

    ccsm4_annual_cntl_aus = get_aus(ccsm4_annual_cntl)          
    save_netcdf_compression(ccsm4_annual_cntl_aus, cntl_output_dir, 'ccsm4_precip_cntl_annual_aus')

    csiro_mk3l_annual_cntl_aus = get_aus(csiro_mk3l_annual_cntl)
    save_netcdf_compression(csiro_mk3l_annual_cntl_aus, cntl_output_dir, 'csiro_mk3l_precip_cntl_annual_aus')

    fgoals_s2_annual_cntl_aus = get_aus(fgoals_s2_annual_cntl)
    save_netcdf_compression(fgoals_s2_annual_cntl_aus, cntl_output_dir, 'fgoals_s2_precip_cntl_annual_aus')

    giss_1_annual_cntl_aus = get_aus(giss_1_annual_cntl)
    save_netcdf_compression(giss_1_annual_cntl_aus, cntl_output_dir, 'giss_1_precip_cntl_annual_aus')

    giss_2_annual_cntl_aus = get_aus(giss_2_annual_cntl)
    save_netcdf_compression(giss_2_annual_cntl_aus, cntl_output_dir, 'giss_2_precip_cntl_annual_aus')

    giss_3_annual_cntl_aus = get_aus(giss_3_annual_cntl)
    save_netcdf_compression(giss_3_annual_cntl_aus, cntl_output_dir, 'giss_3_precip_cntl_annual_aus')

    giss_41_annual_cntl_aus = get_aus(giss_41_annual_cntl)
    save_netcdf_compression(giss_41_annual_cntl_aus, cntl_output_dir, 'giss_41_precip_cntl_annual_aus')

    hadcm3_annual_cntl_aus = get_aus(hadcm3_annual_cntl)
    save_netcdf_compression(hadcm3_annual_cntl_aus, cntl_output_dir, 'hadcm3_precip_cntl_annual_aus')

    ipsl_annual_cntl_aus = get_aus(ipsl_annual_cntl)
    save_netcdf_compression(ipsl_annual_cntl_aus, cntl_output_dir, 'ipsl_precip_cntl_annual_aus')

    miroc_annual_cntl_aus = get_aus(miroc_annual_cntl)          
    save_netcdf_compression(miroc_annual_cntl_aus, cntl_output_dir, 'miroc_precip_cntl_annual_aus')

    mpi_annual_cntl_aus = get_aus(mpi_annual_cntl)
    save_netcdf_compression(mpi_annual_cntl_aus, cntl_output_dir, 'mpi_precip_cntl_annual_aus')

    mri_annual_cntl_aus = get_aus(mri_annual_cntl)
    save_netcdf_compression(mri_annual_cntl_aus, cntl_output_dir, 'mri_precip_cntl_annual_aus')

    cesmlme_annual_cntl_aus = get_aus(cesmlme_annual_cntl)
    save_netcdf_compression(cesmlme_annual_cntl_aus, cntl_output_dir, 'cesmlme_precip_cntl_annual_aus')
else:
    pass

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# -------------------- REGRIDDING ----------
# Turns out we actually need to regrid before subsetting to Australia... 
# otherwise we lose Tasmania! And no one wants that.

# ---- historical files

if regrid_lme_ff_historical_files is True:
    print("... Regridding CESM-LME full-forcing historical files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_hist_output_dir):
        print("...... Creating %s now "  % regridded_hist_output_dir)
        os.makedirs(regridded_hist_output_dir)

    if not os.path.exists(regridded_hist_output_dir_aus):
        print("...... Creating %s now "  % regridded_hist_output_dir_aus)
        os.makedirs(regridded_hist_output_dir_aus)

    # import the GLOBAL files.

    # ------ CESM-LME full forcing files
    ff1_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '1'))
    ff2_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '2'))
    ff3_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '3'))
    ff4_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '4'))
    ff5_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '5'))
    ff6_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '6'))
    ff7_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '7'))
    ff8_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '8'))
    ff9_precip_hist_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '9'))
    ff10_precip_hist_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '10'))
    ff11_precip_hist_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '11'))
    ff12_precip_hist_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '12'))
    ff13_precip_hist_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_hist_annual.nc' % (hist_output_dir + '/global', '13'))

    # calculate drought sum
    ff1_precip_hist_annual   = calculate_drought_sum(ff1_precip_hist_annual)
    ff2_precip_hist_annual   = calculate_drought_sum(ff2_precip_hist_annual)
    ff3_precip_hist_annual   = calculate_drought_sum(ff3_precip_hist_annual)
    ff4_precip_hist_annual   = calculate_drought_sum(ff4_precip_hist_annual)
    ff5_precip_hist_annual   = calculate_drought_sum(ff5_precip_hist_annual)
    ff6_precip_hist_annual   = calculate_drought_sum(ff6_precip_hist_annual)
    ff7_precip_hist_annual   = calculate_drought_sum(ff7_precip_hist_annual)
    ff8_precip_hist_annual   = calculate_drought_sum(ff8_precip_hist_annual)
    ff9_precip_hist_annual   = calculate_drought_sum(ff9_precip_hist_annual)
    ff10_precip_hist_annual  = calculate_drought_sum(ff10_precip_hist_annual)
    ff11_precip_hist_annual  = calculate_drought_sum(ff11_precip_hist_annual)
    ff12_precip_hist_annual  = calculate_drought_sum(ff12_precip_hist_annual)
    ff13_precip_hist_annual  = calculate_drought_sum(ff13_precip_hist_annual)

    # regrid the files
    ff1_precip_hist_annual_rg  = regrid_files(ff1_precip_hist_annual)
    ff2_precip_hist_annual_rg  = regrid_files(ff2_precip_hist_annual)
    ff3_precip_hist_annual_rg  = regrid_files(ff3_precip_hist_annual)
    ff4_precip_hist_annual_rg  = regrid_files(ff4_precip_hist_annual)
    ff5_precip_hist_annual_rg  = regrid_files(ff5_precip_hist_annual)
    ff6_precip_hist_annual_rg  = regrid_files(ff6_precip_hist_annual)
    ff7_precip_hist_annual_rg  = regrid_files(ff7_precip_hist_annual)
    ff8_precip_hist_annual_rg  = regrid_files(ff8_precip_hist_annual)
    ff9_precip_hist_annual_rg  = regrid_files(ff9_precip_hist_annual)
    ff10_precip_hist_annual_rg = regrid_files(ff10_precip_hist_annual)
    ff11_precip_hist_annual_rg = regrid_files(ff11_precip_hist_annual)
    ff12_precip_hist_annual_rg = regrid_files(ff12_precip_hist_annual)
    ff13_precip_hist_annual_rg = regrid_files(ff13_precip_hist_annual)

    # save the file
    save_netcdf_compression(ff1_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '1')
    save_netcdf_compression(ff2_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '2')
    save_netcdf_compression(ff3_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '3')
    save_netcdf_compression(ff4_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '4')
    save_netcdf_compression(ff5_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '5')
    save_netcdf_compression(ff6_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '6')
    save_netcdf_compression(ff7_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '7')
    save_netcdf_compression(ff8_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '8')
    save_netcdf_compression(ff9_precip_hist_annual_rg , regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '9')
    save_netcdf_compression(ff10_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '10')
    save_netcdf_compression(ff11_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '11')
    save_netcdf_compression(ff12_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '12')
    save_netcdf_compression(ff13_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ff%s_precip_hist_annual_2d' % '13')

    print("...... Subsetting regridded files to Australia only")
    # Subset to Australia
    ff1_precip_hist_annual_aus_rg  = get_aus(ff1_precip_hist_annual_rg) 
    ff2_precip_hist_annual_aus_rg  = get_aus(ff2_precip_hist_annual_rg) 
    ff3_precip_hist_annual_aus_rg  = get_aus(ff3_precip_hist_annual_rg) 
    ff4_precip_hist_annual_aus_rg  = get_aus(ff4_precip_hist_annual_rg) 
    ff5_precip_hist_annual_aus_rg  = get_aus(ff5_precip_hist_annual_rg) 
    ff6_precip_hist_annual_aus_rg  = get_aus(ff6_precip_hist_annual_rg) 
    ff7_precip_hist_annual_aus_rg  = get_aus(ff7_precip_hist_annual_rg) 
    ff8_precip_hist_annual_aus_rg  = get_aus(ff8_precip_hist_annual_rg) 
    ff9_precip_hist_annual_aus_rg  = get_aus(ff9_precip_hist_annual_rg) 
    ff10_precip_hist_annual_aus_rg = get_aus(ff10_precip_hist_annual_rg)
    ff11_precip_hist_annual_aus_rg = get_aus(ff11_precip_hist_annual_rg)
    ff12_precip_hist_annual_aus_rg = get_aus(ff12_precip_hist_annual_rg)
    ff13_precip_hist_annual_aus_rg = get_aus(ff13_precip_hist_annual_rg)

    # save files Aus-only regridded version
    save_netcdf_compression(ff1_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '1')
    save_netcdf_compression(ff2_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '2')
    save_netcdf_compression(ff3_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '3')
    save_netcdf_compression(ff4_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '4')
    save_netcdf_compression(ff5_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '5')
    save_netcdf_compression(ff6_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '6')
    save_netcdf_compression(ff7_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '7')
    save_netcdf_compression(ff8_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '8')
    save_netcdf_compression(ff9_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '9')
    save_netcdf_compression(ff10_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '10')
    save_netcdf_compression(ff11_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '11')
    save_netcdf_compression(ff12_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '12')
    save_netcdf_compression(ff13_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ff%s_precip_hist_annual_aus_2d' % '13')

else:
    pass


if regrid_lme_single_forcing_historical_files is True:
    print("... Regridding CESM-LME single-forcing historical files to 2° x 2°")

    # create the output directories
    if not os.path.exists(regridded_hist_output_dir):
        print("...... Creating %s now "  % regridded_hist_output_dir)
        os.makedirs(regridded_hist_output_dir)

    if not os.path.exists(regridded_hist_output_dir_aus):
        print("...... Creating %s now "  % regridded_hist_output_dir_aus)
        os.makedirs(regridded_hist_output_dir_aus)

    # import the GLOBAL files.

     # ------ CESM-LME single forcing files
    # import files
    lme_850forcing3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-850forcing3_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ghg1_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg2_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ghg2_precip_hist_annual.nc' % hist_output_dir)
    lme_ghg3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-ghg3_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_hist_annual.nc' % hist_output_dir)
    lme_lulc3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_hist_annual.nc' % hist_output_dir)
    lme_orbital3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-solar1_precip_hist_annual.nc' % hist_output_dir)
    lme_solar3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-solar3_precip_hist_annual.nc' % hist_output_dir)
    lme_solar4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-solar4_precip_hist_annual.nc' % hist_output_dir)
    lme_solar5_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-solar5_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone1_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone2_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone3_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone4_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_hist_annual.nc' % hist_output_dir)
    lme_ozone5_precip_hist_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_hist_annual.nc' % hist_output_dir)
    lme_volc1_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-volc1_precip_hist_annual.nc' % hist_output_dir)
    lme_volc2_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-volc2_precip_hist_annual.nc' % hist_output_dir)
    lme_volc3_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-volc3_precip_hist_annual.nc' % hist_output_dir)
    lme_volc4_precip_hist_annual  = xr.open_dataset('%s/global/cesmlme-volc4_precip_hist_annual.nc' % hist_output_dir)

    # calculate drought sum
    lme_850forcing3_precip_hist_annual = calculate_drought_sum(lme_850forcing3_precip_hist_annual )
    lme_ghg1_precip_hist_annual   = calculate_drought_sum(lme_ghg1_precip_hist_annual)
    lme_ghg2_precip_hist_annual   = calculate_drought_sum(lme_ghg2_precip_hist_annual)
    lme_ghg3_precip_hist_annual   = calculate_drought_sum(lme_ghg3_precip_hist_annual)
    lme_lulc1_precip_hist_annual   = calculate_drought_sum(lme_lulc1_precip_hist_annual)
    lme_lulc2_precip_hist_annual   = calculate_drought_sum(lme_lulc2_precip_hist_annual)
    lme_lulc3_precip_hist_annual   = calculate_drought_sum(lme_lulc3_precip_hist_annual)
    lme_orbital1_precip_hist_annual  = calculate_drought_sum(lme_orbital1_precip_hist_annual)
    lme_orbital2_precip_hist_annual  = calculate_drought_sum(lme_orbital2_precip_hist_annual)
    lme_orbital3_precip_hist_annual  = calculate_drought_sum(lme_orbital3_precip_hist_annual)
    lme_solar1_precip_hist_annual  = calculate_drought_sum(lme_solar1_precip_hist_annual)
    lme_solar3_precip_hist_annual  = calculate_drought_sum(lme_solar3_precip_hist_annual)
    lme_solar4_precip_hist_annual  = calculate_drought_sum(lme_solar4_precip_hist_annual)
    lme_solar5_precip_hist_annual  = calculate_drought_sum(lme_solar5_precip_hist_annual)
    lme_ozone1_precip_hist_annual  = calculate_drought_sum(lme_ozone1_precip_hist_annual)
    lme_ozone2_precip_hist_annual  = calculate_drought_sum(lme_ozone2_precip_hist_annual)
    lme_ozone3_precip_hist_annual  = calculate_drought_sum(lme_ozone3_precip_hist_annual)
    lme_ozone4_precip_hist_annual  = calculate_drought_sum(lme_ozone4_precip_hist_annual)
    lme_ozone5_precip_hist_annual  = calculate_drought_sum(lme_ozone5_precip_hist_annual)
    lme_volc1_precip_hist_annual  = calculate_drought_sum(lme_volc1_precip_hist_annual)
    lme_volc2_precip_hist_annual  = calculate_drought_sum(lme_volc2_precip_hist_annual)
    lme_volc3_precip_hist_annual  = calculate_drought_sum(lme_volc3_precip_hist_annual)
    lme_volc4_precip_hist_annual  = calculate_drought_sum(lme_volc4_precip_hist_annual)

    # regrid files
    lme_850forcing3_precip_hist_annual_rg  = regrid_files(lme_850forcing3_precip_hist_annual)
    lme_ghg1_precip_hist_annual_rg  = regrid_files(lme_ghg1_precip_hist_annual)
    lme_ghg2_precip_hist_annual_rg  = regrid_files(lme_ghg2_precip_hist_annual)
    lme_ghg3_precip_hist_annual_rg  = regrid_files(lme_ghg3_precip_hist_annual)
    lme_lulc1_precip_hist_annual_rg = regrid_files(lme_lulc1_precip_hist_annual)
    lme_lulc2_precip_hist_annual_rg = regrid_files(lme_lulc2_precip_hist_annual)
    lme_lulc3_precip_hist_annual_rg = regrid_files(lme_lulc3_precip_hist_annual)
    lme_orbital1_precip_hist_annual_rg  = regrid_files(lme_orbital1_precip_hist_annual)
    lme_orbital2_precip_hist_annual_rg  = regrid_files(lme_orbital2_precip_hist_annual)
    lme_orbital3_precip_hist_annual_rg  = regrid_files(lme_orbital3_precip_hist_annual)
    lme_solar1_precip_hist_annual_rg = regrid_files(lme_solar1_precip_hist_annual)
    lme_solar3_precip_hist_annual_rg = regrid_files(lme_solar3_precip_hist_annual)
    lme_solar4_precip_hist_annual_rg = regrid_files(lme_solar4_precip_hist_annual)
    lme_solar5_precip_hist_annual_rg = regrid_files(lme_solar5_precip_hist_annual)
    lme_ozone1_precip_hist_annual_rg = regrid_files(lme_ozone1_precip_hist_annual)
    lme_ozone2_precip_hist_annual_rg = regrid_files(lme_ozone2_precip_hist_annual)
    lme_ozone3_precip_hist_annual_rg = regrid_files(lme_ozone3_precip_hist_annual)
    lme_ozone4_precip_hist_annual_rg = regrid_files(lme_ozone4_precip_hist_annual)
    lme_ozone5_precip_hist_annual_rg = regrid_files(lme_ozone5_precip_hist_annual)
    lme_volc1_precip_hist_annual_rg = regrid_files(lme_volc1_precip_hist_annual)
    lme_volc2_precip_hist_annual_rg = regrid_files(lme_volc2_precip_hist_annual)
    lme_volc3_precip_hist_annual_rg = regrid_files(lme_volc3_precip_hist_annual)
    lme_volc4_precip_hist_annual_rg = regrid_files(lme_volc4_precip_hist_annual)

    # save files
    save_netcdf_compression(lme_850forcing3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-850forcing3_precip_hist_annual_2d')
    save_netcdf_compression(lme_ghg1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ghg1_precip_hist_annual_2d')
    save_netcdf_compression(lme_ghg2_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ghg2_precip_hist_annual_2d')
    save_netcdf_compression(lme_ghg3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ghg3_precip_hist_annual_2d')
    save_netcdf_compression(lme_lulc1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-lulc1_precip_hist_annual_2d')
    save_netcdf_compression(lme_lulc2_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-lulc2_precip_hist_annual_2d')
    save_netcdf_compression(lme_lulc3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-lulc3_precip_hist_annual_2d')
    save_netcdf_compression(lme_orbital1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-orbital1_precip_hist_annual_2d')
    save_netcdf_compression(lme_orbital2_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-orbital2_precip_hist_annual_2d')
    save_netcdf_compression(lme_orbital3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-orbital3_precip_hist_annual_2d')
    save_netcdf_compression(lme_solar1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-solar1_precip_hist_annual_2d')
    save_netcdf_compression(lme_solar3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-solar3_precip_hist_annual_2d')
    save_netcdf_compression(lme_solar4_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-solar4_precip_hist_annual_2d')
    save_netcdf_compression(lme_solar5_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-solar5_precip_hist_annual_2d')
    save_netcdf_compression(lme_ozone1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ozone1_precip_hist_annual_2d')
    save_netcdf_compression(lme_ozone2_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ozone2_precip_hist_annual_2d')
    save_netcdf_compression(lme_ozone3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ozone3_precip_hist_annual_2d')
    save_netcdf_compression(lme_ozone4_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ozone4_precip_hist_annual_2d')
    save_netcdf_compression(lme_ozone5_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-ozone5_precip_hist_annual_2d')
    save_netcdf_compression(lme_volc1_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-volc1_precip_hist_annual_2d')
    save_netcdf_compression(lme_volc2_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-volc2_precip_hist_annual_2d')
    save_netcdf_compression(lme_volc3_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-volc3_precip_hist_annual_2d')
    save_netcdf_compression(lme_volc4_precip_hist_annual_rg, regridded_hist_output_dir, 'cesmlme-volc4_precip_hist_annual_2d')

    print("...... Subsetting regridded files to Australia only")
    # subset to Australia
    lme_850forcing3_precip_hist_annual_aus_rg  = get_aus(lme_850forcing3_precip_hist_annual_rg)
    lme_ghg1_precip_hist_annual_aus_rg  = get_aus(lme_ghg1_precip_hist_annual_rg)
    lme_ghg2_precip_hist_annual_aus_rg  = get_aus(lme_ghg2_precip_hist_annual_rg)
    lme_ghg3_precip_hist_annual_aus_rg  = get_aus(lme_ghg3_precip_hist_annual_rg)
    lme_lulc1_precip_hist_annual_aus_rg = get_aus(lme_lulc1_precip_hist_annual_rg)
    lme_lulc2_precip_hist_annual_aus_rg = get_aus(lme_lulc2_precip_hist_annual_rg)
    lme_lulc3_precip_hist_annual_aus_rg = get_aus(lme_lulc3_precip_hist_annual_rg)
    lme_orbital1_precip_hist_annual_aus_rg  = get_aus(lme_orbital1_precip_hist_annual_rg)
    lme_orbital2_precip_hist_annual_aus_rg  = get_aus(lme_orbital2_precip_hist_annual_rg)
    lme_orbital3_precip_hist_annual_aus_rg  = get_aus(lme_orbital3_precip_hist_annual_rg)
    lme_solar1_precip_hist_annual_aus_rg = get_aus(lme_solar1_precip_hist_annual_rg)
    lme_solar3_precip_hist_annual_aus_rg = get_aus(lme_solar3_precip_hist_annual_rg)
    lme_solar4_precip_hist_annual_aus_rg = get_aus(lme_solar4_precip_hist_annual_rg)
    lme_solar5_precip_hist_annual_aus_rg = get_aus(lme_solar5_precip_hist_annual_rg)
    lme_ozone1_precip_hist_annual_aus_rg = get_aus(lme_ozone1_precip_hist_annual_rg)
    lme_ozone2_precip_hist_annual_aus_rg = get_aus(lme_ozone2_precip_hist_annual_rg)
    lme_ozone3_precip_hist_annual_aus_rg = get_aus(lme_ozone3_precip_hist_annual_rg)
    lme_ozone4_precip_hist_annual_aus_rg = get_aus(lme_ozone4_precip_hist_annual_rg)
    lme_ozone5_precip_hist_annual_aus_rg = get_aus(lme_ozone5_precip_hist_annual_rg)
    lme_volc1_precip_hist_annual_aus_rg = get_aus(lme_volc1_precip_hist_annual_rg)
    lme_volc2_precip_hist_annual_aus_rg = get_aus(lme_volc2_precip_hist_annual_rg)
    lme_volc3_precip_hist_annual_aus_rg = get_aus(lme_volc3_precip_hist_annual_rg)
    lme_volc4_precip_hist_annual_aus_rg = get_aus(lme_volc4_precip_hist_annual_rg)

    # save subsetted regridded files
    save_netcdf_compression(lme_850forcing3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-850forcing3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ghg1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ghg1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ghg2_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ghg2_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ghg3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ghg3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_lulc1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-lulc1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_lulc2_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-lulc2_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_lulc3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-lulc3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_orbital1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-orbital1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_orbital2_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-orbital2_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_orbital3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-orbital3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_solar1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-solar1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_solar3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-solar3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_solar4_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-solar4_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_solar5_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-solar5_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ozone1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ozone1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ozone2_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ozone2_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ozone3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ozone3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ozone4_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ozone4_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_ozone5_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-ozone5_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_volc1_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-volc1_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_volc2_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-volc2_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_volc3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-volc3_precip_hist_annual_aus_2d')
    save_netcdf_compression(lme_volc4_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'cesmlme-volc4_precip_hist_annual_aus_2d')

else:
    pass


if regrid_pmip3_historical_files is True:
    print("... Regridding PMIP3 historical files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_hist_output_dir):
        print("...... Creating %s now "  % regridded_hist_output_dir)
        os.makedirs(regridded_hist_output_dir)

    if not os.path.exists(regridded_hist_output_dir_aus):
        print("...... Creating %s now "  % regridded_hist_output_dir_aus)
        os.makedirs(regridded_hist_output_dir_aus)

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

    # get rid of unneeded bounds
    bcc_precip_hist_annual = bcc_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    ccsm4_precip_hist_annual = ccsm4_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    csiro_mk3l_precip_hist_annual = csiro_mk3l_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    fgoals_gl_precip_hist_annual = fgoals_gl_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    fgoals_s2_precip_hist_annual = fgoals_s2_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_21_precip_hist_annual = giss_21_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_22_precip_hist_annual = giss_22_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_23_precip_hist_annual = giss_23_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_24_precip_hist_annual = giss_24_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_25_precip_hist_annual = giss_25_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_26_precip_hist_annual = giss_26_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_27_precip_hist_annual = giss_27_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_28_precip_hist_annual = giss_28_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    hadcm3_precip_hist_annual = hadcm3_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    ipsl_precip_hist_annual = ipsl_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    miroc_precip_hist_annual = miroc_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    mpi_precip_hist_annual = mpi_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    mri_precip_hist_annual = mri_precip_hist_annual.drop_vars(('lat_bnds', 'lon_bnds'))

    # calculate drought sum
    bcc_precip_hist_annual   = calculate_drought_sum(bcc_precip_hist_annual)
    ccsm4_precip_hist_annual   = calculate_drought_sum(ccsm4_precip_hist_annual)
    csiro_mk3l_precip_hist_annual   = calculate_drought_sum(csiro_mk3l_precip_hist_annual)
    fgoals_gl_precip_hist_annual   = calculate_drought_sum(fgoals_gl_precip_hist_annual)
    fgoals_s2_precip_hist_annual   = calculate_drought_sum(fgoals_s2_precip_hist_annual)
    giss_21_precip_hist_annual   = calculate_drought_sum(giss_21_precip_hist_annual)
    giss_22_precip_hist_annual   = calculate_drought_sum(giss_22_precip_hist_annual)
    giss_23_precip_hist_annual   = calculate_drought_sum(giss_23_precip_hist_annual)
    giss_24_precip_hist_annual   = calculate_drought_sum(giss_24_precip_hist_annual)
    giss_25_precip_hist_annual  = calculate_drought_sum(giss_25_precip_hist_annual)
    giss_26_precip_hist_annual  = calculate_drought_sum(giss_26_precip_hist_annual)
    giss_27_precip_hist_annual  = calculate_drought_sum(giss_27_precip_hist_annual)
    giss_28_precip_hist_annual  = calculate_drought_sum(giss_28_precip_hist_annual)
    hadcm3_precip_hist_annual  = calculate_drought_sum(hadcm3_precip_hist_annual)
    ipsl_precip_hist_annual  = calculate_drought_sum(ipsl_precip_hist_annual)
    miroc_precip_hist_annual  = calculate_drought_sum(miroc_precip_hist_annual)
    mpi_precip_hist_annual  = calculate_drought_sum(mpi_precip_hist_annual)
    mri_precip_hist_annual  = calculate_drought_sum(mri_precip_hist_annual)

    # regrid files
    bcc_precip_hist_annual_rg        = regrid_files(bcc_precip_hist_annual)
    ccsm4_precip_hist_annual_rg      = regrid_files(ccsm4_precip_hist_annual)
    csiro_mk3l_precip_hist_annual_rg = regrid_files(csiro_mk3l_precip_hist_annual)
    fgoals_gl_precip_hist_annual_rg  = regrid_files(fgoals_gl_precip_hist_annual)
    fgoals_s2_precip_hist_annual_rg  = regrid_files(fgoals_s2_precip_hist_annual)
    giss_21_precip_hist_annual_rg    = regrid_files(giss_21_precip_hist_annual)
    giss_22_precip_hist_annual_rg    = regrid_files(giss_22_precip_hist_annual)
    giss_23_precip_hist_annual_rg    = regrid_files(giss_23_precip_hist_annual)
    giss_24_precip_hist_annual_rg    = regrid_files(giss_24_precip_hist_annual)
    giss_25_precip_hist_annual_rg    = regrid_files(giss_25_precip_hist_annual)
    giss_26_precip_hist_annual_rg    = regrid_files(giss_26_precip_hist_annual)
    giss_27_precip_hist_annual_rg    = regrid_files(giss_27_precip_hist_annual)
    giss_28_precip_hist_annual_rg    = regrid_files(giss_28_precip_hist_annual)
    hadcm3_precip_hist_annual_rg     = regrid_files(hadcm3_precip_hist_annual)
    ipsl_precip_hist_annual_rg       = regrid_files(ipsl_precip_hist_annual)
    miroc_precip_hist_annual_rg      = regrid_files(miroc_precip_hist_annual)
    mpi_precip_hist_annual_rg        = regrid_files(mpi_precip_hist_annual)
    mri_precip_hist_annual_rg        = regrid_files(mri_precip_hist_annual)

    # save global regridded files
    save_netcdf_compression(bcc_precip_hist_annual_rg, regridded_hist_output_dir, 'bcc_precip_hist_annual_2d')
    save_netcdf_compression(ccsm4_precip_hist_annual_rg, regridded_hist_output_dir, 'ccsm4_precip_hist_annual_2d')
    save_netcdf_compression(csiro_mk3l_precip_hist_annual_rg, regridded_hist_output_dir, 'csiro_mk3l_precip_hist_annual_2d')
    save_netcdf_compression(fgoals_gl_precip_hist_annual_rg, regridded_hist_output_dir, 'fgoals_gl_precip_hist_annual_2d')
    save_netcdf_compression(fgoals_s2_precip_hist_annual_rg, regridded_hist_output_dir, 'fgoals_s2_precip_hist_annual_2d')
    save_netcdf_compression(giss_21_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_21_precip_hist_annual_2d')
    save_netcdf_compression(giss_22_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_22_precip_hist_annual_2d')
    save_netcdf_compression(giss_23_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_23_precip_hist_annual_2d')
    save_netcdf_compression(giss_24_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_24_precip_hist_annual_2d')
    save_netcdf_compression(giss_25_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_25_precip_hist_annual_2d')
    save_netcdf_compression(giss_26_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_26_precip_hist_annual_2d')
    save_netcdf_compression(giss_27_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_27_precip_hist_annual_2d')
    save_netcdf_compression(giss_28_precip_hist_annual_rg, regridded_hist_output_dir, 'giss_28_precip_hist_annual_2d')
    save_netcdf_compression(hadcm3_precip_hist_annual_rg, regridded_hist_output_dir, 'hadcm3_precip_hist_annual_2d')
    save_netcdf_compression(ipsl_precip_hist_annual_rg, regridded_hist_output_dir, 'ipsl_precip_hist_annual_2d')
    save_netcdf_compression(miroc_precip_hist_annual_rg, regridded_hist_output_dir, 'miroc_precip_hist_annual_2d')
    save_netcdf_compression(mpi_precip_hist_annual_rg, regridded_hist_output_dir, 'mpi_precip_hist_annual_2d')
    save_netcdf_compression(mri_precip_hist_annual_rg, regridded_hist_output_dir, 'mri_precip_hist_annual_2d')

    print("...... Subsetting regridded files to Australia only")
    # subset to Australia
    bcc_precip_hist_annual_aus_rg        = get_aus(bcc_precip_hist_annual_rg)
    ccsm4_precip_hist_annual_aus_rg      = get_aus(ccsm4_precip_hist_annual_rg)
    csiro_mk3l_precip_hist_annual_aus_rg = get_aus(csiro_mk3l_precip_hist_annual_rg)
    fgoals_gl_precip_hist_annual_aus_rg  = get_aus(fgoals_gl_precip_hist_annual_rg)
    fgoals_s2_precip_hist_annual_aus_rg  = get_aus(fgoals_s2_precip_hist_annual_rg)
    giss_21_precip_hist_annual_aus_rg    = get_aus(giss_21_precip_hist_annual_rg)
    giss_22_precip_hist_annual_aus_rg    = get_aus(giss_22_precip_hist_annual_rg)
    giss_23_precip_hist_annual_aus_rg    = get_aus(giss_23_precip_hist_annual_rg)
    giss_24_precip_hist_annual_aus_rg    = get_aus(giss_24_precip_hist_annual_rg)
    giss_25_precip_hist_annual_aus_rg    = get_aus(giss_25_precip_hist_annual_rg)
    giss_26_precip_hist_annual_aus_rg    = get_aus(giss_26_precip_hist_annual_rg)
    giss_27_precip_hist_annual_aus_rg    = get_aus(giss_27_precip_hist_annual_rg)
    giss_28_precip_hist_annual_aus_rg    = get_aus(giss_28_precip_hist_annual_rg)
    hadcm3_precip_hist_annual_aus_rg     = get_aus(hadcm3_precip_hist_annual_rg)
    ipsl_precip_hist_annual_aus_rg       = get_aus(ipsl_precip_hist_annual_rg)
    miroc_precip_hist_annual_aus_rg      = get_aus(miroc_precip_hist_annual_rg)
    mpi_precip_hist_annual_aus_rg        = get_aus(mpi_precip_hist_annual_rg)
    mri_precip_hist_annual_aus_rg        = get_aus(mri_precip_hist_annual_rg)

    # save regridded Aus-only
    save_netcdf_compression(bcc_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'bcc_precip_hist_annual_aus_2d')
    save_netcdf_compression(ccsm4_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'ccsm4_precip_hist_annual_aus_2d')
    save_netcdf_compression(csiro_mk3l_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'csiro_mk3l_precip_hist_annual_aus_2d')
    save_netcdf_compression(fgoals_gl_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'fgoals_gl_precip_hist_annual_aus_2d')
    save_netcdf_compression(fgoals_s2_precip_hist_annual_aus_rg , regridded_hist_output_dir_aus, 'fgoals_s2_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_21_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_21_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_22_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_22_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_23_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_23_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_24_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_24_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_25_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_25_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_26_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_26_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_27_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_27_precip_hist_annual_aus_2d')
    save_netcdf_compression(giss_28_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'giss_28_precip_hist_annual_aus_2d')
    save_netcdf_compression(hadcm3_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'hadcm3_precip_hist_annual_aus_2d')
    save_netcdf_compression(ipsl_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'ipsl_precip_hist_annual_aus_2d')
    save_netcdf_compression(miroc_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'miroc_precip_hist_annual_aus_2d')
    save_netcdf_compression(mpi_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'mpi_precip_hist_annual_aus_2d')
    save_netcdf_compression(mri_precip_hist_annual_aus_rg, regridded_hist_output_dir_aus, 'mri_precip_hist_annual_aus_2d')
    
else:
    pass


# ---- last millennium files
if regrid_lme_ff_lastmillennium_files is True:
    print("... Regridding CESM-LME full-forcing last millennium files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_lm_output_dir):
        print("...... Creating %s now "  % regridded_lm_output_dir)
        os.makedirs(regridded_lm_output_dir)

    if not os.path.exists(regridded_lm_output_dir_aus):
        print("...... Creating %s now "  % regridded_lm_output_dir_aus)
        os.makedirs(regridded_lm_output_dir_aus)

    # import the GLOBAL files.

    # ------ CESM-LME full forcing files
    ff1_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '1'))
    ff2_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '2'))
    ff3_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '3'))
    ff4_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '4'))
    ff5_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '5'))
    ff6_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '6'))
    ff7_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '7'))
    ff8_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '8'))
    ff9_precip_lm_annual  = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '9'))
    ff10_precip_lm_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '10'))
    ff11_precip_lm_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '11'))
    ff12_precip_lm_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '12'))
    ff13_precip_lm_annual = xr.open_dataset('%s/cesmlme-ff%s_precip_lm_annual.nc' % (lm_output_dir + '/global', '13'))

    # calculate drought sum
    ff1_precip_lm_annual   = calculate_drought_sum(ff1_precip_lm_annual)
    ff2_precip_lm_annual   = calculate_drought_sum(ff2_precip_lm_annual)
    ff3_precip_lm_annual   = calculate_drought_sum(ff3_precip_lm_annual)
    ff4_precip_lm_annual   = calculate_drought_sum(ff4_precip_lm_annual)
    ff5_precip_lm_annual   = calculate_drought_sum(ff5_precip_lm_annual)
    ff6_precip_lm_annual   = calculate_drought_sum(ff6_precip_lm_annual)
    ff7_precip_lm_annual   = calculate_drought_sum(ff7_precip_lm_annual)
    ff8_precip_lm_annual   = calculate_drought_sum(ff8_precip_lm_annual)
    ff9_precip_lm_annual   = calculate_drought_sum(ff9_precip_lm_annual)
    ff10_precip_lm_annual  = calculate_drought_sum(ff10_precip_lm_annual)
    ff11_precip_lm_annual  = calculate_drought_sum(ff11_precip_lm_annual)
    ff12_precip_lm_annual  = calculate_drought_sum(ff12_precip_lm_annual)
    ff13_precip_lm_annual  = calculate_drought_sum(ff13_precip_lm_annual)

    # regrid the files
    ff1_precip_lm_annual_rg  = regrid_files(ff1_precip_lm_annual)
    ff2_precip_lm_annual_rg  = regrid_files(ff2_precip_lm_annual)
    ff3_precip_lm_annual_rg  = regrid_files(ff3_precip_lm_annual)
    ff4_precip_lm_annual_rg  = regrid_files(ff4_precip_lm_annual)
    ff5_precip_lm_annual_rg  = regrid_files(ff5_precip_lm_annual)
    ff6_precip_lm_annual_rg  = regrid_files(ff6_precip_lm_annual)
    ff7_precip_lm_annual_rg  = regrid_files(ff7_precip_lm_annual)
    ff8_precip_lm_annual_rg  = regrid_files(ff8_precip_lm_annual)
    ff9_precip_lm_annual_rg  = regrid_files(ff9_precip_lm_annual)
    ff10_precip_lm_annual_rg = regrid_files(ff10_precip_lm_annual)
    ff11_precip_lm_annual_rg = regrid_files(ff11_precip_lm_annual)
    ff12_precip_lm_annual_rg = regrid_files(ff12_precip_lm_annual)
    ff13_precip_lm_annual_rg = regrid_files(ff13_precip_lm_annual)

    # save the file
    save_netcdf_compression(ff1_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '1')
    save_netcdf_compression(ff2_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '2')
    save_netcdf_compression(ff3_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '3')
    save_netcdf_compression(ff4_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '4')
    save_netcdf_compression(ff5_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '5')
    save_netcdf_compression(ff6_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '6')
    save_netcdf_compression(ff7_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '7')
    save_netcdf_compression(ff8_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '8')
    save_netcdf_compression(ff9_precip_lm_annual_rg,  regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '9')
    save_netcdf_compression(ff10_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '10')
    save_netcdf_compression(ff11_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '11')
    save_netcdf_compression(ff12_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '12')
    save_netcdf_compression(ff13_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ff%s_precip_lm_annual_2d' % '13')
    
    print("...... Subsetting regridded files to Australia only")
    # Subset to Australia
    ff1_precip_lm_annual_aus_rg  = get_aus(ff1_precip_lm_annual_rg) 
    ff2_precip_lm_annual_aus_rg  = get_aus(ff2_precip_lm_annual_rg) 
    ff3_precip_lm_annual_aus_rg  = get_aus(ff3_precip_lm_annual_rg) 
    ff4_precip_lm_annual_aus_rg  = get_aus(ff4_precip_lm_annual_rg) 
    ff5_precip_lm_annual_aus_rg  = get_aus(ff5_precip_lm_annual_rg) 
    ff6_precip_lm_annual_aus_rg  = get_aus(ff6_precip_lm_annual_rg) 
    ff7_precip_lm_annual_aus_rg  = get_aus(ff7_precip_lm_annual_rg) 
    ff8_precip_lm_annual_aus_rg  = get_aus(ff8_precip_lm_annual_rg) 
    ff9_precip_lm_annual_aus_rg  = get_aus(ff9_precip_lm_annual_rg) 
    ff10_precip_lm_annual_aus_rg = get_aus(ff10_precip_lm_annual_rg)
    ff11_precip_lm_annual_aus_rg = get_aus(ff11_precip_lm_annual_rg)
    ff12_precip_lm_annual_aus_rg = get_aus(ff12_precip_lm_annual_rg)
    ff13_precip_lm_annual_aus_rg = get_aus(ff13_precip_lm_annual_rg)

    # save files Aus-only regridded version
    save_netcdf_compression(ff1_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '1')
    save_netcdf_compression(ff2_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '2')
    save_netcdf_compression(ff3_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '3')
    save_netcdf_compression(ff4_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '4')
    save_netcdf_compression(ff5_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '5')
    save_netcdf_compression(ff6_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '6')
    save_netcdf_compression(ff7_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '7')
    save_netcdf_compression(ff8_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '8')
    save_netcdf_compression(ff9_precip_lm_annual_aus_rg , regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '9')
    save_netcdf_compression(ff10_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '10')
    save_netcdf_compression(ff11_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '11')
    save_netcdf_compression(ff12_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '12')
    save_netcdf_compression(ff13_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ff%s_precip_lm_annual_aus_2d' % '13')

else:
    pass


if regrid_lme_single_forcing_lastmillennium_files is True:
    print("... Regridding CESM-LME single forcing lasm millennium files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_lm_output_dir):
        print("...... Creating %s now "  % regridded_lm_output_dir)
        os.makedirs(regridded_lm_output_dir)

    if not os.path.exists(regridded_lm_output_dir_aus):
        print("...... Creating %s now "  % regridded_lm_output_dir_aus)
        os.makedirs(regridded_lm_output_dir_aus)

    # import the GLOBAL files.

     # ------ CESM-LME single forcing files
    # import files
    lme_850forcing3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-850forcing3_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ghg1_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg2_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ghg2_precip_lm_annual.nc' % lm_output_dir)
    lme_ghg3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-ghg3_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc1_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc2_precip_lm_annual.nc' % lm_output_dir)
    lme_lulc3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-lulc3_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital1_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital2_precip_lm_annual.nc' % lm_output_dir)
    lme_orbital3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-orbital3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-solar1_precip_lm_annual.nc' % lm_output_dir)
    lme_solar3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-solar3_precip_lm_annual.nc' % lm_output_dir)
    lme_solar4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-solar4_precip_lm_annual.nc' % lm_output_dir)
    lme_solar5_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-solar5_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone1_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone1_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone2_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone2_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone3_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone3_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone4_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone4_precip_lm_annual.nc' % lm_output_dir)
    lme_ozone5_precip_lm_annual = xr.open_dataset('%s/global/cesmlme-ozone5_precip_lm_annual.nc' % lm_output_dir)
    lme_volc1_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-volc1_precip_lm_annual.nc' % lm_output_dir)
    lme_volc2_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-volc2_precip_lm_annual.nc' % lm_output_dir)
    lme_volc3_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-volc3_precip_lm_annual.nc' % lm_output_dir)
    lme_volc4_precip_lm_annual  = xr.open_dataset('%s/global/cesmlme-volc4_precip_lm_annual.nc' % lm_output_dir)

    # calculate drought sum
    lme_850forcing3_precip_lm_annual = calculate_drought_sum(lme_850forcing3_precip_lm_annual )
    lme_ghg1_precip_lm_annual   = calculate_drought_sum(lme_ghg1_precip_lm_annual)
    lme_ghg2_precip_lm_annual   = calculate_drought_sum(lme_ghg2_precip_lm_annual)
    lme_ghg3_precip_lm_annual   = calculate_drought_sum(lme_ghg3_precip_lm_annual)
    lme_lulc1_precip_lm_annual   = calculate_drought_sum(lme_lulc1_precip_lm_annual)
    lme_lulc2_precip_lm_annual   = calculate_drought_sum(lme_lulc2_precip_lm_annual)
    lme_lulc3_precip_lm_annual   = calculate_drought_sum(lme_lulc3_precip_lm_annual)
    lme_orbital1_precip_lm_annual  = calculate_drought_sum(lme_orbital1_precip_lm_annual)
    lme_orbital2_precip_lm_annual  = calculate_drought_sum(lme_orbital2_precip_lm_annual)
    lme_orbital3_precip_lm_annual  = calculate_drought_sum(lme_orbital3_precip_lm_annual)
    lme_solar1_precip_lm_annual  = calculate_drought_sum(lme_solar1_precip_lm_annual)
    lme_solar3_precip_lm_annual  = calculate_drought_sum(lme_solar3_precip_lm_annual)
    lme_solar4_precip_lm_annual  = calculate_drought_sum(lme_solar4_precip_lm_annual)
    lme_solar5_precip_lm_annual  = calculate_drought_sum(lme_solar5_precip_lm_annual)
    lme_ozone1_precip_lm_annual  = calculate_drought_sum(lme_ozone1_precip_lm_annual)
    lme_ozone2_precip_lm_annual  = calculate_drought_sum(lme_ozone2_precip_lm_annual)
    lme_ozone3_precip_lm_annual  = calculate_drought_sum(lme_ozone3_precip_lm_annual)
    lme_ozone4_precip_lm_annual  = calculate_drought_sum(lme_ozone4_precip_lm_annual)
    lme_ozone5_precip_lm_annual  = calculate_drought_sum(lme_ozone5_precip_lm_annual)
    lme_volc1_precip_lm_annual  = calculate_drought_sum(lme_volc1_precip_lm_annual)
    lme_volc2_precip_lm_annual  = calculate_drought_sum(lme_volc2_precip_lm_annual)
    lme_volc3_precip_lm_annual  = calculate_drought_sum(lme_volc3_precip_lm_annual)
    lme_volc4_precip_lm_annual  = calculate_drought_sum(lme_volc4_precip_lm_annual)

    # regrid files
    lme_850forcing3_precip_lm_annual_rg  = regrid_files(lme_850forcing3_precip_lm_annual)
    lme_ghg1_precip_lm_annual_rg  = regrid_files(lme_ghg1_precip_lm_annual)
    lme_ghg2_precip_lm_annual_rg  = regrid_files(lme_ghg2_precip_lm_annual)
    lme_ghg3_precip_lm_annual_rg  = regrid_files(lme_ghg3_precip_lm_annual)
    lme_lulc1_precip_lm_annual_rg = regrid_files(lme_lulc1_precip_lm_annual)
    lme_lulc2_precip_lm_annual_rg = regrid_files(lme_lulc2_precip_lm_annual)
    lme_lulc3_precip_lm_annual_rg = regrid_files(lme_lulc3_precip_lm_annual)
    lme_orbital1_precip_lm_annual_rg  = regrid_files(lme_orbital1_precip_lm_annual)
    lme_orbital2_precip_lm_annual_rg  = regrid_files(lme_orbital2_precip_lm_annual)
    lme_orbital3_precip_lm_annual_rg  = regrid_files(lme_orbital3_precip_lm_annual)
    lme_solar1_precip_lm_annual_rg = regrid_files(lme_solar1_precip_lm_annual)
    lme_solar3_precip_lm_annual_rg = regrid_files(lme_solar3_precip_lm_annual)
    lme_solar4_precip_lm_annual_rg = regrid_files(lme_solar4_precip_lm_annual)
    lme_solar5_precip_lm_annual_rg = regrid_files(lme_solar5_precip_lm_annual)
    lme_ozone1_precip_lm_annual_rg = regrid_files(lme_ozone1_precip_lm_annual)
    lme_ozone2_precip_lm_annual_rg = regrid_files(lme_ozone2_precip_lm_annual)
    lme_ozone3_precip_lm_annual_rg = regrid_files(lme_ozone3_precip_lm_annual)
    lme_ozone4_precip_lm_annual_rg = regrid_files(lme_ozone4_precip_lm_annual)
    lme_ozone5_precip_lm_annual_rg = regrid_files(lme_ozone5_precip_lm_annual)
    lme_volc1_precip_lm_annual_rg = regrid_files(lme_volc1_precip_lm_annual)
    lme_volc2_precip_lm_annual_rg = regrid_files(lme_volc2_precip_lm_annual)
    lme_volc3_precip_lm_annual_rg = regrid_files(lme_volc3_precip_lm_annual)
    lme_volc4_precip_lm_annual_rg = regrid_files(lme_volc4_precip_lm_annual)

    # save files
    save_netcdf_compression(lme_850forcing3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-850forcing3_precip_lm_annual_2d')
    save_netcdf_compression(lme_ghg1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ghg1_precip_lm_annual_2d')
    save_netcdf_compression(lme_ghg2_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ghg2_precip_lm_annual_2d')
    save_netcdf_compression(lme_ghg3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ghg3_precip_lm_annual_2d')
    save_netcdf_compression(lme_lulc1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-lulc1_precip_lm_annual_2d')
    save_netcdf_compression(lme_lulc2_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-lulc2_precip_lm_annual_2d')
    save_netcdf_compression(lme_lulc3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-lulc3_precip_lm_annual_2d')
    save_netcdf_compression(lme_orbital1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-orbital1_precip_lm_annual_2d')
    save_netcdf_compression(lme_orbital2_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-orbital2_precip_lm_annual_2d')
    save_netcdf_compression(lme_orbital3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-orbital3_precip_lm_annual_2d')
    save_netcdf_compression(lme_solar1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-solar1_precip_lm_annual_2d')
    save_netcdf_compression(lme_solar3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-solar3_precip_lm_annual_2d')
    save_netcdf_compression(lme_solar4_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-solar4_precip_lm_annual_2d')
    save_netcdf_compression(lme_solar5_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-solar5_precip_lm_annual_2d')
    save_netcdf_compression(lme_ozone1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ozone1_precip_lm_annual_2d')
    save_netcdf_compression(lme_ozone2_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ozone2_precip_lm_annual_2d')
    save_netcdf_compression(lme_ozone3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ozone3_precip_lm_annual_2d')
    save_netcdf_compression(lme_ozone4_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ozone4_precip_lm_annual_2d')
    save_netcdf_compression(lme_ozone5_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-ozone5_precip_lm_annual_2d')
    save_netcdf_compression(lme_volc1_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-volc1_precip_lm_annual_2d')
    save_netcdf_compression(lme_volc2_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-volc2_precip_lm_annual_2d')
    save_netcdf_compression(lme_volc3_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-volc3_precip_lm_annual_2d')
    save_netcdf_compression(lme_volc4_precip_lm_annual_rg, regridded_lm_output_dir, 'cesmlme-volc4_precip_lm_annual_2d')
    
    print("...... Subsetting regridded files to Australia only")
    # subset to Australia
    lme_850forcing3_precip_lm_annual_aus_rg  = get_aus(lme_850forcing3_precip_lm_annual_rg)
    lme_ghg1_precip_lm_annual_aus_rg  = get_aus(lme_ghg1_precip_lm_annual_rg)
    lme_ghg2_precip_lm_annual_aus_rg  = get_aus(lme_ghg2_precip_lm_annual_rg)
    lme_ghg3_precip_lm_annual_aus_rg  = get_aus(lme_ghg3_precip_lm_annual_rg)
    lme_lulc1_precip_lm_annual_aus_rg = get_aus(lme_lulc1_precip_lm_annual_rg)
    lme_lulc2_precip_lm_annual_aus_rg = get_aus(lme_lulc2_precip_lm_annual_rg)
    lme_lulc3_precip_lm_annual_aus_rg = get_aus(lme_lulc3_precip_lm_annual_rg)
    lme_orbital1_precip_lm_annual_aus_rg  = get_aus(lme_orbital1_precip_lm_annual_rg)
    lme_orbital2_precip_lm_annual_aus_rg  = get_aus(lme_orbital2_precip_lm_annual_rg)
    lme_orbital3_precip_lm_annual_aus_rg  = get_aus(lme_orbital3_precip_lm_annual_rg)
    lme_solar1_precip_lm_annual_aus_rg = get_aus(lme_solar1_precip_lm_annual_rg)
    lme_solar3_precip_lm_annual_aus_rg = get_aus(lme_solar3_precip_lm_annual_rg)
    lme_solar4_precip_lm_annual_aus_rg = get_aus(lme_solar4_precip_lm_annual_rg)
    lme_solar5_precip_lm_annual_aus_rg = get_aus(lme_solar5_precip_lm_annual_rg)
    lme_ozone1_precip_lm_annual_aus_rg = get_aus(lme_ozone1_precip_lm_annual_rg)
    lme_ozone2_precip_lm_annual_aus_rg = get_aus(lme_ozone2_precip_lm_annual_rg)
    lme_ozone3_precip_lm_annual_aus_rg = get_aus(lme_ozone3_precip_lm_annual_rg)
    lme_ozone4_precip_lm_annual_aus_rg = get_aus(lme_ozone4_precip_lm_annual_rg)
    lme_ozone5_precip_lm_annual_aus_rg = get_aus(lme_ozone5_precip_lm_annual_rg)
    lme_volc1_precip_lm_annual_aus_rg = get_aus(lme_volc1_precip_lm_annual_rg)
    lme_volc2_precip_lm_annual_aus_rg = get_aus(lme_volc2_precip_lm_annual_rg)
    lme_volc3_precip_lm_annual_aus_rg = get_aus(lme_volc3_precip_lm_annual_rg)
    lme_volc4_precip_lm_annual_aus_rg = get_aus(lme_volc4_precip_lm_annual_rg)

    # save subsetted regridded files
    save_netcdf_compression(lme_850forcing3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-850forcing3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ghg1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ghg1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ghg2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ghg2_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ghg3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ghg3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_lulc1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-lulc1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_lulc2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-lulc2_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_lulc3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-lulc3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_orbital1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-orbital1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_orbital2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-orbital2_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_orbital3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-orbital3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_solar1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-solar1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_solar3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-solar3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_solar4_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-solar4_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_solar5_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-solar5_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ozone1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ozone1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ozone2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ozone2_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ozone3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ozone3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ozone4_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ozone4_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_ozone5_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-ozone5_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_volc1_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-volc1_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_volc2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-volc2_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_volc3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-volc3_precip_lm_annual_aus_2d')
    save_netcdf_compression(lme_volc4_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'cesmlme-volc4_precip_lm_annual_aus_2d')

else:
    pass


if regrid_pmip3_lastmillennium_files is True:
    print("... Regridding PMIP3 last millennium files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_lm_output_dir):
        print("...... Creating %s now "  % regridded_lm_output_dir)
        os.makedirs(regridded_lm_output_dir)

    if not os.path.exists(regridded_lm_output_dir_aus):
        print("...... Creating %s now "  % regridded_lm_output_dir_aus)
        os.makedirs(regridded_lm_output_dir_aus)

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
    hadcm3_precip_lm_annual     = xr.open_dataset('%s/global/hadcm3_precip_lm_annual.nc' % lm_output_dir)
    ipsl_precip_lm_annual       = xr.open_dataset('%s/global/ipsl_precip_lm_annual.nc' % lm_output_dir)
    miroc_precip_lm_annual      = xr.open_dataset('%s/global/miroc_precip_lm_annual.nc' % lm_output_dir)
    mpi_precip_lm_annual        = xr.open_dataset('%s/global/mpi_precip_lm_annual.nc' % lm_output_dir)
    mri_precip_lm_annual        = xr.open_dataset('%s/global/mri_precip_lm_annual.nc' % lm_output_dir)

    # get rid of unneeded bounds
    bcc_precip_lm_annual = bcc_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    ccsm4_precip_lm_annual = ccsm4_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    csiro_mk3l_precip_lm_annual = csiro_mk3l_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    fgoals_gl_precip_lm_annual = fgoals_gl_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    fgoals_s2_precip_lm_annual = fgoals_s2_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_21_precip_lm_annual = giss_21_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_22_precip_lm_annual = giss_22_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_23_precip_lm_annual = giss_23_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_24_precip_lm_annual = giss_24_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_25_precip_lm_annual = giss_25_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_26_precip_lm_annual = giss_26_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_27_precip_lm_annual = giss_27_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_28_precip_lm_annual = giss_28_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    hadcm3_precip_lm_annual = hadcm3_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    ipsl_precip_lm_annual = ipsl_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    miroc_precip_lm_annual = miroc_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    mpi_precip_lm_annual = mpi_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))
    mri_precip_lm_annual = mri_precip_lm_annual.drop_vars(('lat_bnds', 'lon_bnds'))

    # calculate drought sum
    bcc_precip_lm_annual   = calculate_drought_sum(bcc_precip_lm_annual)
    ccsm4_precip_lm_annual   = calculate_drought_sum(ccsm4_precip_lm_annual)
    csiro_mk3l_precip_lm_annual   = calculate_drought_sum(csiro_mk3l_precip_lm_annual)
    fgoals_gl_precip_lm_annual   = calculate_drought_sum(fgoals_gl_precip_lm_annual)
    fgoals_s2_precip_lm_annual   = calculate_drought_sum(fgoals_s2_precip_lm_annual)
    giss_21_precip_lm_annual = calculate_drought_sum(giss_21_precip_lm_annual)
    giss_22_precip_lm_annual = calculate_drought_sum(giss_22_precip_lm_annual)
    giss_23_precip_lm_annual = calculate_drought_sum(giss_23_precip_lm_annual)
    giss_24_precip_lm_annual = calculate_drought_sum(giss_24_precip_lm_annual)
    giss_25_precip_lm_annual = calculate_drought_sum(giss_25_precip_lm_annual)
    giss_26_precip_lm_annual = calculate_drought_sum(giss_26_precip_lm_annual)
    giss_27_precip_lm_annual = calculate_drought_sum(giss_27_precip_lm_annual)
    giss_28_precip_lm_annual = calculate_drought_sum(giss_28_precip_lm_annual)
    hadcm3_precip_lm_annual  = calculate_drought_sum(hadcm3_precip_lm_annual)
    ipsl_precip_lm_annual  = calculate_drought_sum(ipsl_precip_lm_annual)
    miroc_precip_lm_annual  = calculate_drought_sum(miroc_precip_lm_annual)
    mpi_precip_lm_annual  = calculate_drought_sum(mpi_precip_lm_annual)
    mri_precip_lm_annual  = calculate_drought_sum(mri_precip_lm_annual)

    # regrid files
    bcc_precip_lm_annual_rg        = regrid_files(bcc_precip_lm_annual)
    ccsm4_precip_lm_annual_rg      = regrid_files(ccsm4_precip_lm_annual)
    csiro_mk3l_precip_lm_annual_rg = regrid_files(csiro_mk3l_precip_lm_annual)
    fgoals_gl_precip_lm_annual_rg  = regrid_files(fgoals_gl_precip_lm_annual)
    fgoals_s2_precip_lm_annual_rg  = regrid_files(fgoals_s2_precip_lm_annual)
    giss_21_precip_lm_annual_rg    = regrid_files(giss_21_precip_lm_annual)
    giss_22_precip_lm_annual_rg    = regrid_files(giss_22_precip_lm_annual)
    giss_23_precip_lm_annual_rg    = regrid_files(giss_23_precip_lm_annual)
    giss_24_precip_lm_annual_rg    = regrid_files(giss_24_precip_lm_annual)
    giss_25_precip_lm_annual_rg    = regrid_files(giss_25_precip_lm_annual)
    giss_26_precip_lm_annual_rg    = regrid_files(giss_26_precip_lm_annual)
    giss_27_precip_lm_annual_rg    = regrid_files(giss_27_precip_lm_annual)
    giss_28_precip_lm_annual_rg    = regrid_files(giss_28_precip_lm_annual)
    hadcm3_precip_lm_annual_rg     = regrid_files(hadcm3_precip_lm_annual)
    ipsl_precip_lm_annual_rg       = regrid_files(ipsl_precip_lm_annual)
    miroc_precip_lm_annual_rg      = regrid_files(miroc_precip_lm_annual)
    mpi_precip_lm_annual_rg        = regrid_files(mpi_precip_lm_annual)
    mri_precip_lm_annual_rg        = regrid_files(mri_precip_lm_annual)

    # save global regridded files
    save_netcdf_compression(bcc_precip_lm_annual_rg, regridded_lm_output_dir, 'bcc_precip_lm_annual_2d')
    save_netcdf_compression(ccsm4_precip_lm_annual_rg, regridded_lm_output_dir, 'ccsm4_precip_lm_annual_2d')
    save_netcdf_compression(csiro_mk3l_precip_lm_annual_rg, regridded_lm_output_dir, 'csiro_mk3l_precip_lm_annual_2d')
    save_netcdf_compression(fgoals_gl_precip_lm_annual_rg, regridded_lm_output_dir, 'fgoals_gl_precip_lm_annual_2d')
    save_netcdf_compression(fgoals_s2_precip_lm_annual_rg, regridded_lm_output_dir, 'fgoals_s2_precip_lm_annual_2d')
    save_netcdf_compression(giss_21_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_21_precip_lm_annual_2d')
    save_netcdf_compression(giss_22_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_22_precip_lm_annual_2d')
    save_netcdf_compression(giss_23_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_23_precip_lm_annual_2d')
    save_netcdf_compression(giss_24_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_24_precip_lm_annual_2d')
    save_netcdf_compression(giss_25_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_25_precip_lm_annual_2d')
    save_netcdf_compression(giss_26_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_26_precip_lm_annual_2d')
    save_netcdf_compression(giss_27_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_27_precip_lm_annual_2d')
    save_netcdf_compression(giss_28_precip_lm_annual_rg, regridded_lm_output_dir, 'giss_28_precip_lm_annual_2d')
    save_netcdf_compression(hadcm3_precip_lm_annual_rg, regridded_lm_output_dir, 'hadcm3_precip_lm_annual_2d')
    save_netcdf_compression(ipsl_precip_lm_annual_rg, regridded_lm_output_dir, 'ipsl_precip_lm_annual_2d')
    save_netcdf_compression(miroc_precip_lm_annual_rg, regridded_lm_output_dir, 'miroc_precip_lm_annual_2d')
    save_netcdf_compression(mpi_precip_lm_annual_rg, regridded_lm_output_dir, 'mpi_precip_lm_annual_2d')
    save_netcdf_compression(mri_precip_lm_annual_rg, regridded_lm_output_dir, 'mri_precip_lm_annual_2d')
    
    print("...... Subsetting regridded files to Australia only")
    # subset to Australia
    bcc_precip_lm_annual_aus_rg        = get_aus(bcc_precip_lm_annual_rg)
    ccsm4_precip_lm_annual_aus_rg      = get_aus(ccsm4_precip_lm_annual_rg)
    csiro_mk3l_precip_lm_annual_aus_rg = get_aus(csiro_mk3l_precip_lm_annual_rg)
    fgoals_gl_precip_lm_annual_aus_rg  = get_aus(fgoals_gl_precip_lm_annual_rg)
    fgoals_s2_precip_lm_annual_aus_rg  = get_aus(fgoals_s2_precip_lm_annual_rg)
    giss_21_precip_lm_annual_aus_rg    = get_aus(giss_21_precip_lm_annual_rg)
    giss_22_precip_lm_annual_aus_rg    = get_aus(giss_22_precip_lm_annual_rg)
    giss_23_precip_lm_annual_aus_rg    = get_aus(giss_23_precip_lm_annual_rg)
    giss_24_precip_lm_annual_aus_rg    = get_aus(giss_24_precip_lm_annual_rg)
    giss_25_precip_lm_annual_aus_rg    = get_aus(giss_25_precip_lm_annual_rg)
    giss_26_precip_lm_annual_aus_rg    = get_aus(giss_26_precip_lm_annual_rg)
    giss_27_precip_lm_annual_aus_rg    = get_aus(giss_27_precip_lm_annual_rg)
    giss_28_precip_lm_annual_aus_rg    = get_aus(giss_28_precip_lm_annual_rg)
    hadcm3_precip_lm_annual_aus_rg     = get_aus(hadcm3_precip_lm_annual_rg)
    ipsl_precip_lm_annual_aus_rg       = get_aus(ipsl_precip_lm_annual_rg)
    miroc_precip_lm_annual_aus_rg      = get_aus(miroc_precip_lm_annual_rg)
    mpi_precip_lm_annual_aus_rg        = get_aus(mpi_precip_lm_annual_rg)
    mri_precip_lm_annual_aus_rg        = get_aus(mri_precip_lm_annual_rg)

    # save regridded Aus-only
    save_netcdf_compression(bcc_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'bcc_precip_lm_annual_aus_2d')
    save_netcdf_compression(ccsm4_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'ccsm4_precip_lm_annual_aus_2d')
    save_netcdf_compression(csiro_mk3l_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'csiro_mk3l_precip_lm_annual_aus_2d')
    save_netcdf_compression(fgoals_gl_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'fgoals_gl_precip_lm_annual_aus_2d')
    save_netcdf_compression(fgoals_s2_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'fgoals_s2_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_21_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_21_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_22_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_22_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_23_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_23_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_24_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_24_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_25_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_25_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_26_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_26_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_27_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_27_precip_lm_annual_aus_2d')
    save_netcdf_compression(giss_28_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'giss_28_precip_lm_annual_aus_2d')
    save_netcdf_compression(hadcm3_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'hadcm3_precip_lm_annual_aus_2d')
    save_netcdf_compression(ipsl_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'ipsl_precip_lm_annual_aus_2d')
    save_netcdf_compression(miroc_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'miroc_precip_lm_annual_aus_2d')
    save_netcdf_compression(mpi_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'mpi_precip_lm_annual_aus_2d')
    save_netcdf_compression(mri_precip_lm_annual_aus_rg, regridded_lm_output_dir_aus, 'mri_precip_lm_annual_aus_2d')
    
else:
    pass

# ---- control files
if regrid_control_files is True:
    print("... Regridding control files to 2° x 2°")
    # create the output directories
    if not os.path.exists(regridded_cntl_output_dir):
        print("...... Creating %s now "  % regridded_cntl_output_dir)
        os.makedirs(regridded_cntl_output_dir)

    if not os.path.exists(regridded_cntl_output_dir_aus):
        print("...... Creating %s now "  % regridded_cntl_output_dir_aus)
        os.makedirs(regridded_cntl_output_dir_aus)

    bcc_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'bcc_precip_cntl_annual.nc'))
    ccsm4_cntl  = xr.open_dataset('%s/%s' % (cntl_output_dir, 'ccsm4_precip_cntl_annual.nc'))
    csiro_mk3l_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'csiro_mk3l_precip_cntl_annual.nc'))
    fgoals_s2_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'fgoals_s2_precip_cntl_annual.nc'))
    giss_2_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'giss_2_precip_cntl_annual.nc'))
    giss_1_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'giss_1_precip_cntl_annual.nc'))
    giss_3_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'giss_3_precip_cntl_annual.nc'))
    giss_41_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'giss_41_precip_cntl_annual.nc'))
    hadmc3_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'hadcm3_precip_cntl_annual.nc'))
    ipsl_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'ipsl_precip_cntl_annual.nc'))
    miroc_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'miroc_precip_cntl_annual.nc'))
    mpi_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'mpi_precip_cntl_annual.nc'))
    mri_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'mri_precip_cntl_annual.nc'))
    cesmlme_cntl = xr.open_dataset('%s/%s' % (cntl_output_dir, 'cesmlme_precip_cntl_annual.nc'))

    # get rid of unneeded bounds
    bcc_cntl = bcc_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    ccsm4_cntl = ccsm4_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    csiro_mk3l_cntl = csiro_mk3l_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    fgoals_s2_cntl = fgoals_s2_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_1_cntl = giss_1_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_2_cntl = giss_2_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_3_cntl = giss_3_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    giss_41_cntl = giss_41_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    hadmc3_cntl = hadmc3_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    ipsl_cntl = ipsl_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    miroc_cntl = miroc_cntl.drop_vars(('lat_bnds', 'lon_bnds'))
    mpi_cntl = mpi_cntl.drop_vars(('lat_bnds', 'lon_bnds', 'time_bnds'))
    mri_cntl = mri_cntl.drop_vars(('lat_bnds', 'lon_bnds'))

    # calculate drought sum
    bcc_cntl = calculate_drought_sum(bcc_cntl)
    ccsm4_cntl = calculate_drought_sum(ccsm4_cntl)
    csiro_mk3l_cntl = calculate_drought_sum(csiro_mk3l_cntl)
    fgoals_s2_cntl = calculate_drought_sum(fgoals_s2_cntl)
    giss_1_cntl = calculate_drought_sum(giss_1_cntl)
    giss_2_cntl = calculate_drought_sum(giss_2_cntl)
    giss_3_cntl = calculate_drought_sum(giss_3_cntl)
    giss_41_cntl = calculate_drought_sum(giss_41_cntl)
    hadmc3_cntl = calculate_drought_sum(hadmc3_cntl)
    ipsl_cntl = calculate_drought_sum(ipsl_cntl)
    miroc_cntl = calculate_drought_sum(miroc_cntl)
    mpi_cntl = calculate_drought_sum(mpi_cntl)
    mri_cntl = calculate_drought_sum(mri_cntl)
    cesmlme_cntl = calculate_drought_sum(cesmlme_cntl)

    # regrid files
    bcc_cntl_rg = regrid_files(bcc_cntl)
    ccsm4_cntl_rg   = regrid_files(ccsm4_cntl)
    csiro_mk3l_cntl_rg = regrid_files(csiro_mk3l_cntl)
    fgoals_s2_cntl_rg = regrid_files(fgoals_s2_cntl)
    giss_1_cntl_rg = regrid_files(giss_1_cntl)
    giss_2_cntl_rg = regrid_files(giss_2_cntl)
    giss_3_cntl_rg = regrid_files(giss_3_cntl)
    giss_41_cntl_rg = regrid_files(giss_41_cntl)
    hadmc3_cntl_rg = regrid_files(hadmc3_cntl)
    ipsl_cntl_rg = regrid_files(ipsl_cntl)
    miroc_cntl_rg = regrid_files(miroc_cntl)
    mpi_cntl_rg = regrid_files(mpi_cntl)
    mri_cntl_rg = regrid_files(mri_cntl)
    cesmlme_cntl_rg = regrid_files(cesmlme_cntl)

    # save global regridded files
    save_netcdf_compression(bcc_cntl_rg, regridded_cntl_output_dir, 'bcc_precip_cntl_annual_rg')
    save_netcdf_compression(ccsm4_cntl_rg, regridded_cntl_output_dir, 'ccsm4_precip_cntl_annual_rg')
    save_netcdf_compression(csiro_mk3l_cntl_rg, regridded_cntl_output_dir, 'csiro_mk3l_precip_cntl_annual_rg')
    save_netcdf_compression(fgoals_s2_cntl_rg, regridded_cntl_output_dir, 'fgoals_s2_precip_cntl_annual_rg')
    save_netcdf_compression(giss_1_cntl_rg, regridded_cntl_output_dir, 'giss_1_precip_cntl_annual_rg')
    save_netcdf_compression(giss_2_cntl_rg, regridded_cntl_output_dir, 'giss_2_precip_cntl_annual_rg')
    save_netcdf_compression(giss_3_cntl_rg, regridded_cntl_output_dir, 'giss_3_precip_cntl_annual_rg')
    save_netcdf_compression(giss_41_cntl_rg, regridded_cntl_output_dir, 'giss_41_precip_cntl_annual_rg')
    save_netcdf_compression(hadmc3_cntl_rg, regridded_cntl_output_dir, 'hadcm3_precip_cntl_annual_rg')
    save_netcdf_compression(ipsl_cntl_rg, regridded_cntl_output_dir, 'ipsl_precip_cntl_annual_rg')
    save_netcdf_compression(miroc_cntl_rg, regridded_cntl_output_dir, 'miroc_precip_cntl_annual_rg')
    save_netcdf_compression(mpi_cntl_rg, regridded_cntl_output_dir, 'mpi_precip_cntl_annual_rg')
    save_netcdf_compression(mri_cntl_rg, regridded_cntl_output_dir, 'mri_precip_cntl_annual_rg')
    save_netcdf_compression(cesmlme_cntl_rg, regridded_cntl_output_dir, 'cesmlme_precip_cntl_annual_rg')

    
    print("...... Subsetting regridded files to Australia only")
    # subset to Australia
    bcc_annual_cntl_aus = get_aus(bcc_cntl_rg)
    ccsm4_annual_cntl_aus = get_aus(ccsm4_cntl_rg)
    csiro_mk3l_annual_cntl_aus = get_aus(csiro_mk3l_cntl_rg)
    fgoals_s2_annual_cntl_aus = get_aus(fgoals_s2_cntl_rg)
    giss_1_annual_cntl_aus = get_aus(giss_1_cntl_rg)
    giss_2_annual_cntl_aus = get_aus(giss_2_cntl_rg)
    giss_3_annual_cntl_aus = get_aus(giss_3_cntl_rg)
    giss_41_annual_cntl_aus = get_aus(giss_41_cntl_rg)
    hadcm3_annual_cntl_aus = get_aus(hadmc3_cntl_rg)
    ipsl_annual_cntl_aus = get_aus(ipsl_cntl_rg)
    miroc_annual_cntl_aus = get_aus(miroc_cntl_rg)
    mpi_annual_cntl_aus = get_aus(mpi_cntl_rg)
    mri_annual_cntl_aus = get_aus(mri_cntl_rg)
    cesmlme_cntl_rg_aus = get_aus(cesmlme_cntl_rg)

    # save regridded Aus-only
    save_netcdf_compression(bcc_annual_cntl_aus, regridded_cntl_output_dir_aus, 'bcc_precip_cntl_annual_aus_2d')
    save_netcdf_compression(ccsm4_annual_cntl_aus, regridded_cntl_output_dir_aus, 'ccsm4_precip_cntl_annual_aus_2d')
    save_netcdf_compression(csiro_mk3l_annual_cntl_aus, regridded_cntl_output_dir_aus, 'csiro_mk3l_precip_cntl_annual_aus_2d')
    save_netcdf_compression(fgoals_s2_annual_cntl_aus, regridded_cntl_output_dir_aus, 'fgoals_s2_precip_cntl_annual_aus_2d')
    save_netcdf_compression(giss_1_annual_cntl_aus, regridded_cntl_output_dir_aus, 'giss_1_precip_cntl_annual_aus_2d')
    save_netcdf_compression(giss_2_annual_cntl_aus, regridded_cntl_output_dir_aus, 'giss_2_precip_cntl_annual_aus_2d')
    save_netcdf_compression(giss_3_annual_cntl_aus, regridded_cntl_output_dir_aus, 'giss_3_precip_cntl_annual_aus_2d')
    save_netcdf_compression(giss_41_annual_cntl_aus, regridded_cntl_output_dir_aus, 'giss_41_precip_cntl_annual_aus_2d')
    save_netcdf_compression(hadcm3_annual_cntl_aus, regridded_cntl_output_dir_aus, 'hadcm3_precip_cntl_annual_rg')
    save_netcdf_compression(ipsl_annual_cntl_aus, regridded_cntl_output_dir_aus, 'ipsl_precip_cntl_annual_aus_2d')
    save_netcdf_compression(miroc_annual_cntl_aus, regridded_cntl_output_dir_aus, 'miroc_precip_cntl_annual_aus_2d')
    save_netcdf_compression(mpi_annual_cntl_aus, regridded_cntl_output_dir_aus, 'mpi_precip_cntl_annual_aus_2d')
    save_netcdf_compression(mri_annual_cntl_aus, regridded_cntl_output_dir_aus, 'mri_precip_cntl_annual_aus_2d')
    save_netcdf_compression(cesmlme_cntl_rg_aus, regridded_cntl_output_dir_aus, 'cesmlme_precip_cntl_annual_aus_2d')
 
else:
    pass


















