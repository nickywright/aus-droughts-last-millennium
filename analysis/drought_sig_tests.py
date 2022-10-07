## Regrid awap and do significance testing on files
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

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# some options for running
# ---- set output directories etc
historical_year = 1900
hist_output_dir = '../files/historical_1900'

# climatology for lm files
lm_threshold_startyear = 1900
lm_threshold_endyear = 2000

lm_output_dir = '../files/lastmillennium_threshold_1900-2000'

# %% ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# ------ regrid AWAP to other grid resolution
# this is in preparation of doing some stats tests
def save_netcdf_compression(ds, output_dir, filename):

    comp = dict(zlib=True, complevel=8)
    encoding = {var: comp for var in ds.data_vars}

    delayed_obj = ds.to_netcdf('%s/%s.nc' % (output_dir, filename), mode='w', compute=False, encoding=encoding)
    with ProgressBar():
        results = delayed_obj.compute()

# --- DEF
# Instructions for regridding using xesmf are here: https://xesmf.readthedocs.io/en/latest/notebooks/Dataset.html
def regrid_files(ds_to_regrid, ds_target ):
    # resolution of output: same as cesm-lme
    ds_out = xr.Dataset({'lat': (['lat'], ds_target.lat.data),
                         'lon': (['lon'], ds_target.lon.data)})

    regridder = xe.Regridder(ds_to_regrid, ds_out, 'bilinear')
    # regridder.clean_weight_file()

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
print('... importing Aus historical files')
bcc_precip_hist_annual_aus        = xr.open_dataset('%s/aus/bcc_precip_hist_annual_aus.nc' % hist_output_dir)
ccsm4_precip_hist_annual_aus      = xr.open_dataset('%s/aus/ccsm4_precip_hist_annual_aus.nc' % hist_output_dir)
csiro_mk3l_precip_hist_annual_aus = xr.open_dataset('%s/aus/csiro_mk3l_precip_hist_annual_aus.nc' % hist_output_dir)
fgoals_gl_precip_hist_annual_aus  = xr.open_dataset('%s/aus/fgoals_gl_precip_hist_annual_aus.nc' % hist_output_dir)
fgoals_s2_precip_hist_annual_aus  = xr.open_dataset('%s/aus/fgoals_s2_precip_hist_annual_aus.nc' % hist_output_dir)
giss_21_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_21_precip_hist_annual_aus.nc' % hist_output_dir)
giss_22_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_22_precip_hist_annual_aus.nc' % hist_output_dir)
giss_23_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_23_precip_hist_annual_aus.nc' % hist_output_dir)
giss_24_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_24_precip_hist_annual_aus.nc' % hist_output_dir)
giss_25_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_25_precip_hist_annual_aus.nc' % hist_output_dir)
giss_26_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_26_precip_hist_annual_aus.nc' % hist_output_dir)
giss_27_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_27_precip_hist_annual_aus.nc' % hist_output_dir)
giss_28_precip_hist_annual_aus    = xr.open_dataset('%s/aus/giss_28_precip_hist_annual_aus.nc' % hist_output_dir)
hadcm3_precip_hist_annual_aus     = xr.open_dataset('%s/aus/hadcm3_precip_hist_annual_aus.nc' % hist_output_dir)
ipsl_precip_hist_annual_aus       = xr.open_dataset('%s/aus/ipsl_precip_hist_annual_aus.nc' % hist_output_dir)
miroc_precip_hist_annual_aus      = xr.open_dataset('%s/aus/miroc_precip_hist_annual_aus.nc' % hist_output_dir)
mpi_precip_hist_annual_aus        = xr.open_dataset('%s/aus/mpi_precip_hist_annual_aus.nc' % hist_output_dir)
mri_precip_hist_annual_aus        = xr.open_dataset('%s/aus/mri_precip_hist_annual_aus.nc' % hist_output_dir)
giss_all_precip_hist_annual_aus = xr.open_dataset('%s/aus/giss_all_precip_hist_annual_aus.nc' % hist_output_dir)

ff1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff1_precip_hist_annual_aus.nc' % hist_output_dir)
ff2_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff2_precip_hist_annual_aus.nc' % hist_output_dir)
ff3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff3_precip_hist_annual_aus.nc' % hist_output_dir)
ff4_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff4_precip_hist_annual_aus.nc' % hist_output_dir)
ff5_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff5_precip_hist_annual_aus.nc' % hist_output_dir)
ff6_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff6_precip_hist_annual_aus.nc' % hist_output_dir)
ff7_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff7_precip_hist_annual_aus.nc' % hist_output_dir)
ff8_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff8_precip_hist_annual_aus.nc' % hist_output_dir)
ff9_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff9_precip_hist_annual_aus.nc' % hist_output_dir)
ff10_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff10_precip_hist_annual_aus.nc' % hist_output_dir)
ff11_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff11_precip_hist_annual_aus.nc' % hist_output_dir)
ff12_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff12_precip_hist_annual_aus.nc' % hist_output_dir)
ff13_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff13_precip_hist_annual_aus.nc' % hist_output_dir)

ff_all_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff_all_precip_hist_annual_aus.nc' % hist_output_dir)

# read in processed files
lme_850forcing3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-850forcing3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ghg1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ghg2_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg2_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ghg3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_lulc1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_lulc2_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc2_precip_hist_annual_aus.nc' % hist_output_dir)
lme_lulc3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_orbital1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_orbital2_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital2_precip_hist_annual_aus.nc' % hist_output_dir)
lme_orbital3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_solar1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_solar3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_solar4_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar4_precip_hist_annual_aus.nc' % hist_output_dir)
lme_solar5_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar5_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone1_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone2_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone2_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone3_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone4_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone4_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone5_precip_hist_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone5_precip_hist_annual_aus.nc' % hist_output_dir)
lme_volc1_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc1_precip_hist_annual_aus.nc' % hist_output_dir)
lme_volc2_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc2_precip_hist_annual_aus.nc' % hist_output_dir)
lme_volc3_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc3_precip_hist_annual_aus.nc' % hist_output_dir)
lme_volc4_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc4_precip_hist_annual_aus.nc' % hist_output_dir)

lme_ghg_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-ghg_all_precip_hist_annual_aus.nc' % hist_output_dir)
lme_lulc_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-lulc_all_precip_hist_annual_aus.nc' % hist_output_dir)
lme_orbital_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-orbital_all_precip_hist_annual_aus.nc' % hist_output_dir)
lme_solar_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-solar_all_precip_hist_annual_aus.nc' % hist_output_dir)
lme_ozone_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-ozone_all_precip_hist_annual_aus.nc' % hist_output_dir)
lme_volc_all_precip_hist_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc_all_precip_hist_annual_aus.nc' % hist_output_dir)

# ---- last millennium
# read in processed files
bcc_precip_lm_annual_aus        = xr.open_dataset('%s/aus/bcc_precip_lm_annual_aus.nc' % lm_output_dir)
ccsm4_precip_lm_annual_aus      = xr.open_dataset('%s/aus/ccsm4_precip_lm_annual_aus.nc' % lm_output_dir)
csiro_mk3l_precip_lm_annual_aus = xr.open_dataset('%s/aus/csiro_mk3l_precip_lm_annual_aus.nc' % lm_output_dir)
fgoals_gl_precip_lm_annual_aus  = xr.open_dataset('%s/aus/fgoals_gl_precip_lm_annual_aus.nc' % lm_output_dir)
fgoals_s2_precip_lm_annual_aus  = xr.open_dataset('%s/aus/fgoals_s2_precip_lm_annual_aus.nc' % lm_output_dir)
giss_21_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_21_precip_lm_annual_aus.nc' % lm_output_dir)
giss_22_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_22_precip_lm_annual_aus.nc' % lm_output_dir)
giss_23_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_23_precip_lm_annual_aus.nc' % lm_output_dir)
giss_24_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_24_precip_lm_annual_aus.nc' % lm_output_dir)
giss_25_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_25_precip_lm_annual_aus.nc' % lm_output_dir)
giss_26_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_26_precip_lm_annual_aus.nc' % lm_output_dir)
giss_27_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_27_precip_lm_annual_aus.nc' % lm_output_dir)
giss_28_precip_lm_annual_aus    = xr.open_dataset('%s/aus/giss_28_precip_lm_annual_aus.nc' % lm_output_dir)
hadcm3_precip_lm_annual_aus     = xr.open_dataset('%s/aus/hadcm3_precip_lm_annual_aus.nc' % lm_output_dir)
ipsl_precip_lm_annual_aus       = xr.open_dataset('%s/aus/ipsl_precip_lm_annual_aus.nc' % lm_output_dir)
miroc_precip_lm_annual_aus      = xr.open_dataset('%s/aus/miroc_precip_lm_annual_aus.nc' % lm_output_dir)
mpi_precip_lm_annual_aus        = xr.open_dataset('%s/aus/mpi_precip_lm_annual_aus.nc' % lm_output_dir)
mri_precip_lm_annual_aus        = xr.open_dataset('%s/aus/mri_precip_lm_annual_aus.nc' % lm_output_dir)
giss_all_precip_lm_annual_aus   = xr.open_dataset('%s/aus/giss_all_precip_lm_annual_aus.nc' % lm_output_dir)


# import files
ff1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff1_precip_lm_annual_aus.nc' % lm_output_dir)
ff2_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff2_precip_lm_annual_aus.nc' % lm_output_dir)
ff3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff3_precip_lm_annual_aus.nc' % lm_output_dir)
ff4_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff4_precip_lm_annual_aus.nc' % lm_output_dir)
ff5_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff5_precip_lm_annual_aus.nc' % lm_output_dir)
ff6_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff6_precip_lm_annual_aus.nc' % lm_output_dir)
ff7_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff7_precip_lm_annual_aus.nc' % lm_output_dir)
ff8_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff8_precip_lm_annual_aus.nc' % lm_output_dir)
ff9_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff9_precip_lm_annual_aus.nc' % lm_output_dir)
ff10_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff10_precip_lm_annual_aus.nc' % lm_output_dir)
ff11_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff11_precip_lm_annual_aus.nc' % lm_output_dir)
ff12_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff12_precip_lm_annual_aus.nc' % lm_output_dir)
ff13_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff13_precip_lm_annual_aus.nc' % lm_output_dir)

ff_all_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ff_all_precip_lm_annual_aus.nc' % lm_output_dir)


# read in processed files
lme_850forcing3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-850forcing3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ghg1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ghg2_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg2_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ghg3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ghg3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_lulc1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_lulc2_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc2_precip_lm_annual_aus.nc' % lm_output_dir)
lme_lulc3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-lulc3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_orbital1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_orbital2_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital2_precip_lm_annual_aus.nc' % lm_output_dir)
lme_orbital3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-orbital3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_solar1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_solar3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_solar4_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar4_precip_lm_annual_aus.nc' % lm_output_dir)
lme_solar5_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-solar5_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone1_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone2_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone2_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone3_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone4_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone4_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone5_precip_lm_annual_aus = xr.open_dataset('%s/aus/cesmlme-ozone5_precip_lm_annual_aus.nc' % lm_output_dir)
lme_volc1_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc1_precip_lm_annual_aus.nc' % lm_output_dir)
lme_volc2_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc2_precip_lm_annual_aus.nc' % lm_output_dir)
lme_volc3_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc3_precip_lm_annual_aus.nc' % lm_output_dir)
lme_volc4_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc4_precip_lm_annual_aus.nc' % lm_output_dir)

lme_ghg_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-ghg_all_precip_lm_annual_aus.nc' % lm_output_dir)
lme_lulc_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-lulc_all_precip_lm_annual_aus.nc' % lm_output_dir)
lme_orbital_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-orbital_all_precip_lm_annual_aus.nc' % lm_output_dir)
lme_solar_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-solar_all_precip_lm_annual_aus.nc' % lm_output_dir)
lme_ozone_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-ozone_all_precip_lm_annual_aus.nc' % lm_output_dir)
lme_volc_all_precip_lm_annual_aus  = xr.open_dataset('%s/aus/cesmlme-volc_all_precip_lm_annual_aus.nc' % lm_output_dir)





# -----------------------------------------------
# regrid AWAP to whatever model res is
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


