import xarray as xr
import numpy as np


# no longer needed - this is in climate_xr_funcs
# def import_fullforcing_precip(case_no):
#     ff_precip = climate_xr_funcs.import_full_forcing_variable_cam(filepath, case_no, 'PRECT')
#     # convert precip from m/s into mm/month
#     month_length = xr.DataArray(climate_xr_funcs.get_dpm(ff_precip, calendar='noleap'),
#                                 coords=[ff_precip.time], name='month_length')
#     ff_precip['PRECT_mm'] = ff_precip.PRECT * 1000 * 60 * 60 * 24 * month_length
#     return ff_precip


# ---- Drought definition: 2S2E
""" Definition from Coats et al. (2013):
- drought start: 2 consecutive years below mean (starts on 1st year below)
- drought ends: 2 consecutive years above mean (ends before the above mean years)
"""

def get_drought_years_2S2E(ds, threshold):
    """
    Function to loop through years and identify drought years
    this is not the most elegant way to do this, but it works. 
    This will identify drought years based on the definition of Coats et al. (2013):
        - drought start: 2 consecutive years below mean (starts on 1st year below)
        - drought ends: 2 consecutive years above mean (ends before the above mean years)
    
    Usage:  call the 'apply' version (e.g. get_drought_years_2S2E_apply)
            e.g.: ds['drought_years_2S2E'] = get_drought_years_2S2Eapply(ds.PRECT_mm)

    input:  ds[precip] <--- Must specify the precip variable name.
                            Precipitation can be as actual precip, or in anomaly space
     
    output: years for drought (as '1'), everything else is '0'
    """
    
    # OLD
    # --- first check to see if file is in anomaly space or normal precip space
    # assume it's already in anomaly space if there negative precip 
    # (because otherwise we can't have negative precip)
    
    # check to see if threshold is a flaot or array
    # if threshold.min() < 0:
    #     mean_precip = 0
    # else:
    #     mean_precip = threshold.mean()
    
    mean_precip = threshold
        
    # --- get years that are 2 consecutive years below or above mean precip
    start =[]
    end = []
    for i in range(len(ds) - 1):
        # start a drought when there are 2 consecutive years below mean precip
        if ds[i] <= mean_precip and ds[i+1] <= mean_precip:
            start.append(i)
            start.append(i+1)
        # a drought can end after 2 consecutive years above mean precip
        if ds[i] > mean_precip and ds[i+1] > mean_precip:
            end.append(i)
            end.append(i+1)
        
        
    # --- get years above and below mean precip
    ds_below_mean_precip = ds <= mean_precip
    # print(ds_below_mean_precip)
    ds_below_mean_precip = xr.where(ds_below_mean_precip == True, 5, -5)

    # --- if it's a year that drought can start in / is drought, change to 1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_start = np.zeros_like(ds)        # create empty array the same size
    ds_start[start] = 1                 # change indices that are listed in 'start' to 1

    # combine with precip from before, where ...
    # --- definite drought year ==  1
    # --- below mean precip (but may or may not be a drought) == 5
    # --- above mean precip (but may or may not be a drought) == -5
    drought_start = xr.where(ds_start == 1, 1, ds_below_mean_precip)

    # --- if it's a year that drought should end / not a drought, change to -1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_end = np.zeros_like(ds)          # create empty array the same size
    ds_end[end] = -1                    # change indices that are listed in 'end' to -1
    # combine with precip from before, where definite non-drought year is '-999'
    drought_end = xr.where(ds_end == -1, -999, ds_below_mean_precip)

    # combine drought start and end arrays.
    # --- definite drought year ==  1
    # --- definitely not a drought == -999
    # --- below mean precip (but may or may not be a drought) == 5
    # --- above mean precip (but may or may not be a drought) == -5
    drought_poss = xr.where(drought_start > 0, drought_start, drought_end)

    # --- second pass, get more droughts!
    years = []
    for i in range(len(drought_poss) - 2):
        # check if it is definitely a drought year, followed by a year that has negative precip anom
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == 5):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 1):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 5):
            years.append(i+1)
            years.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years] = 1
    drought_poss2 = xr.where(ds_years == 1, 1, drought_poss)

    # --- repeat process
    years2 = []
    for i in range(len(drought_poss2) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == 5):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 1):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 5):
            years2.append(i+1)
            years2.append(i+2)
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years2] = 1
    drought_poss3 = xr.where(ds_years == 1, 1, drought_poss2)

    # --- repeat process
    years3 = []
    for i in range(len(drought_poss3) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == 5):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 1):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 5):
            years3.append(i+1)
            years3.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years3] = 1
    drought_poss4 = xr.where(ds_years == 1, 1, drought_poss3)

    # --- repeat process again (just in case)
    years4 = []
    for i in range(len(drought_poss4) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == 5):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 1):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 5):
            years4.append(i+1)
            years4.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years4] = 1
    drought_poss5 = xr.where(ds_years == 1, 1, drought_poss4)

    # --- convert into drought years vs non-drought years
    drought_years = xr.where(drought_poss5 == 1, 1, 0)

    return drought_years


def get_drought_years_2S2E_gufunc(in_array, threshold, axis):
    # reshape array to a 2d (lat-lon pair, time)
    data = in_array.reshape(axis, in_array.shape[axis])
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        # this might be useful if we don't want to use mean?
        thresholds = np.quantile(in_array, threshold, axis=axis)   
    else:
        # threshold is already an array
        thresholds = threshold
        
    # flatten threshold array
    ds_threshold = thresholds.flatten()
    
    # Iterate through data and threshold, and apply func
    # NOTE: before, e.g.
    #     results = np.apply_along_axis(get_drought_years_below_threshold, 1, data, threshold)
    # is only able to handle 1 input array (for the axis transformation)
    # but np.apply_along_axis is essentially a wrapper for a for-loop
    
    results =[]
    for i in range(data.shape[0]):
        results.append(get_drought_years_2S2E(data[i,:], ds_threshold[i]))
    
    # actually do the drought index procedure
    # results = np.apply_along_axis(get_drought_years_2S2E, 1, data)

    # reshape to original lat, lon, year shape.
    results_reshaped = np.reshape(results, (in_array.shape[0], in_array.shape[1], in_array.shape[2]))
    return results_reshaped


def get_drought_years_2S2E_apply(ds_precip, threshold):
    # e.g. run on ds.PRECT_mm
    drought_years = xr.apply_ufunc(get_drought_years_2S2E_gufunc, ds_precip, threshold, input_core_dims=[['year'], []],
                          output_dtypes=(float), output_core_dims=[['year']], kwargs={'axis': -1})
    # reorder output so it's consistent with input order
    return drought_years.transpose('year', 'lat', 'lon')


# ---- Drought definition: below median --- use threshold version instead
# def get_drought_years_below_median(ds):
#     # droughts are years below median
#     ds_annual_median = ds.median(dim='year')
#     # get precip value relative to the median
#     ds_rel_to_median = ds.PRECT_mm - ds_annual_median.PRECT_mm
#     # count cumulative years below the median (negative rel to median).
#     # restart count when year is above the median.
#     ds_less_than_median = ds_rel_to_median < 0
#
#     # convert to 1s and 0s?
#     ds_drought_years = np.zeros_like(ds.PRECT_mm)
#     ds_drought_years[ds_less_than_median] = 1
#
#     return (('year', 'lat', 'lon'), ds_drought_years)


# ---- Drought definition: below percentile
def get_drought_years_below_threshold(ds, threshold):
    """ Get [drought] years equal to or below a particular threshold
    
    Usage:  call the 'apply' version (e.g. get_drought_years_below_threshold_apply)
            e.g.: ds['drought_years_threshold'] = get_drought_years_below_threshold_apply(ds.PRECT_mm, threshold value)

    input:  ds[precip] <--- Must specify the precip variable name.
                            Precipitation can be as actual precip, or in anomaly space
            threshold  <--- This is a single value (float) - either passed in or calculated within these defs
     
    output: years for drought (as '1'), everything else is '0'
    
    """
    
    # get precip value relative to the threshold
    ds_rel_to_threshold = ds - threshold
    

    # count cumulative years below the median (negative rel to median).
    # restart count when year is above the median.
    # ds_less_than_threshold = ds_rel_to_threshold < 0

    # --- get years that are 2 consecutive years below threshold
    start = []
    end = []
    for i in range(len(ds) - 1):
        # start a drought when there are 2 consecutive years below mean precip
        if ds[i] <= threshold and ds[i+1] <= threshold:
            start.append(i)
            start.append(i+1)

    # convert to 1s and 0s
    ds_drought_years = np.zeros_like(ds)
    ds_drought_years[start] = 1
    
    return ds_drought_years


def get_drought_years_below_threshold_gufunc(in_array, threshold, axis):
    """
    Reshape data and apply the actual 'get_drought_years...' function on a grid-cell basis.
    `threshold` can be:
            - an input array (i.e. a pre-calculated threshold)
            - a single float: in this case, the threshold-array will be calculated across the entire timeseries 
    """
    
    # reshape array to a 2d (lat-lon pair, time)
    data = in_array.reshape(axis, in_array.shape[axis])
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        thresholds = np.quantile(in_array, threshold, axis=axis)   
    else:
        # threshold is already an array
        thresholds = threshold
        
    # flatten threshold array
    ds_threshold = thresholds.flatten()
    
    # Iterate through data and threshold, and apply func
    # NOTE: before, e.g.
    #     results = np.apply_along_axis(get_drought_years_below_threshold, 1, data, threshold)
    # is only able to handle 1 input array (for the axis transformation)
    # but np.apply_along_axis is essentially a wrapper for a for-loop
    
    results =[]
    for i in range(data.shape[0]):
        results.append(get_drought_years_below_threshold(data[i,:], ds_threshold[i]))
        
    # reshape to original lat, lon, year shape.
    results_reshaped = np.reshape(results, (in_array.shape[0], in_array.shape[1], in_array.shape[2]))
    return results_reshaped


def get_drought_years_below_threshold_apply(ds_precip, threshold):
    # e.g. run on ds.PRECT_mm
    # threshold can either be an 2D-array or a single float
    drought_years = xr.apply_ufunc(get_drought_years_below_threshold_gufunc, ds_precip, threshold, input_core_dims=[['year'], []],
                          output_dtypes=(float), output_core_dims=[['year']], kwargs={'axis': -1})
    # reorder output so it's consistent with input order
    return drought_years.transpose('year', 'lat', 'lon')

# ----- Drought defintions: generic

def max_length_ufunc(in_array, dim):
    return xr.apply_ufunc(np.nanmax, in_array, input_core_dims=[[dim]], kwargs={'axis':-1})


def mean_length_ufunc(in_array, dim):
    return xr.apply_ufunc(np.nanmean, in_array, input_core_dims=[[dim]], kwargs={'axis':-1})



def cumulative_drought_length(ds):
    # input: ds[DROUGHT YEAR NAME]
    # this gets the cumulative length of the drought

    # add an empty dimension to the end - this is because xr.diff reduces the dimension by 1
    ds_single_yr = xr.zeros_like(ds[-1])
    ds_single_yr['year'] = ds.year[-1] + 1  # change the year to be 1 after end of original ds
    ds_new = xr.concat([ds, ds_single_yr], dim='year')


    ds_less = ds_new > 0
    ds_cumulative = ds_less.cumsum(dim='year') - ds_less.cumsum(dim='year').where(~ds_less).ffill(dim='year').fillna(0).astype(int)  # ffill = forward fill
    # get rid of intermediate cumulative years
    # (e.g. if a drought is 5 years, get rid of values 1-4, otherwise we'll count them twice later on)
    cumsum_diff = ds_cumulative.diff(dim='year', n=1, label='lower')
#     df_cumulative_max = xr.where(cumsum_diff < 0, cumsum_diff * -1, 0)
    # convert the 0 to NaNs
    drought_length_cumulative = xr.where(cumsum_diff < 0, cumsum_diff * -1, np.nan)


    return drought_length_cumulative

def std_apply(in_array, dim):
    return xr.apply_ufunc(np.nanstd, in_array, input_core_dims=[[dim]], kwargs={'axis':-1})

def sum_apply(in_array, dim):
    return xr.apply_ufunc(np.nansum, in_array, input_core_dims=[[dim]], kwargs={'axis':-1})

# ---- calculate number of individiual drought events, agnostic of length of drought
def count_drought_events(ds):
    return np.count_nonzero(~np.isnan(ds))

def count_drought_events_gufunc(in_array, axis):
    # reshape array to a 2d (lat-lon pair, time)
    data = in_array.reshape(axis, in_array.shape[axis])

    # get autocorrelation for lag1
    results = np.apply_along_axis(count_drought_events, 1, data)

    # reshape to original lat, lon shape.
    # output array should be the size of lat-lon array only
    # e.g. we no longer have a time dimensions
    return results.reshape((in_array.shape[0], in_array.shape[1]))

def count_drought_events_apply(ds):
    return xr.apply_ufunc(count_drought_events_gufunc, ds, input_core_dims=[['year']],
                          output_dtypes=(float),
                          kwargs={'axis': -1})


def drought_intensity(df, drought_type_years, drought_type_len, precip_threshold):
    # precip_threshold: what drought is relative to (e.g. mean, median, quantile...)
    # drought_type_years = 0 and 1s for drought years
    # drought_type_len: overall length of each drought
    
    precip_anom =  df.PRECT_mm - precip_threshold
    # get precip anomaly for drought years only
    df_precip_anom = df[drought_type_years] * precip_anom
    
    # make true/false for years of drought (can't do the invert ('~') bit for the next line with int)
    df_drought_years_true = df[drought_type_years] == 1
    
    # get cumulative precip loss for each drought per year
    df_cumulative = df_precip_anom.cumsum(dim='year') - df_precip_anom.cumsum(dim='year').where(~df_drought_years_true).ffill(dim='year').fillna(0)  # ffill = forward fill

    # get cumulative precip loss for each drought - only using the last year (max cumuative loss)
    df_cumulative_loss = df_cumulative * (df[drought_type_len] / df[drought_type_len])
    df_intensity = df_cumulative_loss / df[drought_type_len]
    return df_intensity
    
def drought_severity(df, drought_type_years, drought_type_len, precip_threshold):
    # precip_threshold: what drought is relative to (e.g. mean, median, quantile...)
    # drought_type_years = 0 and 1s for drought years
    # drought_type_len: overall length of each drought
    
    precip_anom =  df.PRECT_mm - precip_threshold
    # get precip anomaly for drought years only
    df_precip_anom = df[drought_type_years] * precip_anom
    
    # make true/false for years of drought (can't do the invert ('~') bit for the next line with int)
    df_drought_years_true = df[drought_type_years] == 1
    
    # get cumulative precip loss for each drought per year
    df_cumulative = df_precip_anom.cumsum(dim='year') - df_precip_anom.cumsum(dim='year').where(~df_drought_years_true).ffill(dim='year').fillna(0)  # ffill = forward fill

    # get cumulative precip loss for each drought - only using the last year (max cumuative loss)
    df_cumulative_loss = df_cumulative * (df[drought_type_len] / df[drought_type_len])
    #     df_intensity = df_cumulative_loss / df[drought_type_len]
    return df_cumulative_loss


# ---- Drought definition: essentially a modified version of 2S2E.
# Drought starts whenever precip is below 20th perc (doesn't HAVE to be 2 years below 20). E.g. can be 1 x 20% and 1 x 40 % for a drought.
# Drought ends with 2 years above 50th percentile
# drought minimum length is still 2 years

def get_drought_years_120perc_2median(ds, threshold_low, threshold_high):

    
    threshold_20perc = threshold_low
    threshold_50perc = threshold_high
    # print('threshold_20', threshold_20perc)    
    # --- get years that are a single years below 20% and second year below 50%
    start =[]
    end = []
    for i in range(len(ds) - 1):
        # start a drought when the first year is below 20% and second year is below 50%
        if ds[i] <= threshold_20perc and ds[i+1] <= threshold_50perc:
            start.append(i)
            start.append(i+1)
        # a drought can end after 2 consecutive years above mean precip
        if ds[i] > threshold_50perc and ds[i+1] > threshold_50perc:
            end.append(i)
            end.append(i+1)
            
    # --- get years above and below mean precip
    ds_below_median_precip = ds <= threshold_50perc
    # print(ds_below_mean_precip)
    ds_below_median_precip = xr.where(ds_below_median_precip == True, 5, -5)
       
    # --- if it's a year that drought can start in / is drought, change to 1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_start = np.zeros_like(ds)        # create empty array the same size
    ds_start[start] = 1                 # change indices that are listed in 'start' to 1
    
    # combine with precip from before, where ...
    # --- definite drought year ==  1
    # --- below mean precip (but may or may not be a drought) == 5
    # --- above mean precip (but may or may not be a drought) == -5
    drought_start = xr.where(ds_start == 1, 1, ds_below_median_precip)

    # --- if it's a year that drought should end / not a drought, change to -1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_end = np.zeros_like(ds)          # create empty array the same size
    ds_end[end] = -1                    # change indices that are listed in 'end' to -1
    # combine with precip from before, where definite non-drought year is '-999'
    drought_end = xr.where(ds_end == -1, -999, ds_below_median_precip)

    # combine drought start and end arrays.
    # --- definite drought year ==  1
    # --- definitely not a drought == -999
    # --- below mean precip (but may or may not be a drought) == 5
    # --- above mean precip (but may or may not be a drought) == -5
    drought_poss = xr.where(drought_start > 0, drought_start, drought_end)
    
    # --- second pass, get more droughts!
    years = []
    for i in range(len(drought_poss) - 2):
        # check if it is definitely a drought year, followed by a year that has negative precip anom
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == 5):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 1):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 5):
            years.append(i+1)
            years.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years] = 1
    drought_poss2 = xr.where(ds_years == 1, 1, drought_poss)

    # --- repeat process
    years2 = []
    for i in range(len(drought_poss2) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == 5):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 1):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 5):
            years2.append(i+1)
            years2.append(i+2)
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years2] = 1
    drought_poss3 = xr.where(ds_years == 1, 1, drought_poss2)

    # --- repeat process
    years3 = []
    for i in range(len(drought_poss3) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == 5):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 1):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 5):
            years3.append(i+1)
            years3.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years3] = 1
    drought_poss4 = xr.where(ds_years == 1, 1, drought_poss3)

    # --- repeat process again (just in case)
    years4 = []
    for i in range(len(drought_poss4) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == 5):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 1):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 5):
            years4.append(i+1)
            years4.append(i+2)

    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years4] = 1
    drought_poss5 = xr.where(ds_years == 1, 1, drought_poss4)

    # --- convert into drought years vs non-drought years
    drought_years = xr.where(drought_poss5 == 1, 1, 0)
    
    return drought_years
    
    
def get_drought_years_120perc_2median_gufunc(in_array, threshold_start, threshold_end, axis):
    # reshape array to a 2d (lat-lon pair, time)
    data = in_array.reshape(axis, in_array.shape[axis])
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold_start) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        threshold_starts = np.quantile(in_array, threshold_start, axis=axis)   
    else:
        # threshold is already an array
        threshold_starts = threshold_start
        
    # flatten threshold array
    ds_threshold_starts = threshold_starts.flatten()
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold_end) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        threshold_ends = np.quantile(in_array, threshold_end, axis=axis)   
    else:
        # threshold is already an array
        threshold_ends = threshold_end
        
    # flatten threshold array
    ds_threshold_ends = threshold_ends.flatten()
    
    # Iterate through data and threshold, and apply func
    # NOTE: before, e.g.
    #     results = np.apply_along_axis(get_drought_years_below_threshold, 1, data, threshold)
    # is only able to handle 1 input array (for the axis transformation)
    # but np.apply_along_axis is essentially a wrapper for a for-loop
    
    results =[]
    for i in range(data.shape[0]):
        results.append(get_drought_years_120perc_2median(data[i,:], ds_threshold_starts[i], ds_threshold_ends[i]))
        
    # reshape to original lat, lon, year shape.
    results_reshaped = np.reshape(results, (in_array.shape[0], in_array.shape[1], in_array.shape[2]))
    return results_reshaped


def get_drought_years_120perc_2median_apply(ds_precip, threshold_low, threshold_high):
    # e.g. run on ds.PRECT_mm
    drought_years = xr.apply_ufunc(get_drought_years_120perc_2median_gufunc, ds_precip, threshold_low, threshold_high, input_core_dims=[['year'], [], []],
                          output_dtypes=(float), output_core_dims=[['year']], kwargs={'axis': -1})
    # reorder output so it's consistent with input order
    return drought_years.transpose('year', 'lat', 'lon')

    
# ---- Drought definition: below percentile
def get_drought_years_below_start_end_thresholds(ds, threshold_start, threshold_end):
    """ Get [drought] years equal to or below a particular threshold
    Here, the threshold for starting a drought can be different to ending it
    Assumes lower threshold to start a drought (e.g. below 20% to start, above 50% to end)
    
    To start a drought: 2 consecutive years below start threshold (e.g. 20%)
    To end a drought: whenever drought is above end threshold (e.g. 50%)
    NOTE: There will be no individual years of the drought above the end threshold
    
    Usage:  call the 'apply' version (e.g. get_drought_years_below_threshold_apply)
            e.g.: ds['drought_years_threshold'] = get_drought_years_below_threshold_apply(ds.PRECT_mm, threshold value)

    input:  ds[precip] <--- Must specify the precip variable name.
                            Precipitation can be as actual precip, or in anomaly space
            threshold  <--- This is a single value (float) - either passed in or calculated within these defs
     
    output: years for drought (as '1'), everything else is '0'
    
    """
    
    # count cumulative years below the median (negative rel to median).
    # restart count when year is above the median.
    # ds_less_than_threshold = ds_rel_to_threshold < 0

    # threshold_start = ds.quantile(0.2).values
    # threshold_end = ds.quantile(0.5).values
    
    # count cumulative years below the median (negative rel to median).
    # restart count when year is above the median.
    # ds_less_than_threshold = ds_rel_to_threshold < 0
    
    # --- get years that are 2 consecutive years below threshold
    start = []
    end = []
    for i in range(len(ds) - 1):
        # start a drought when there is 2 consecutive years below mean precip
        if ds[i] <= threshold_start and ds[i+1] <= threshold_start:
            start.append(i)
            start.append(i+1)
               # a drought can end after 2 consecutive years above mean precip
        if ds[i] > threshold_end:
            end.append(i)
    
            
    # --- get years above and below mean precip
    ds_below_median_precip = ds <= threshold_end
    # print(ds_below_mean_precip)
    ds_below_median_precip = xr.where(ds_below_median_precip == True, 5, -5)
    
    # --- if it's a year that drought can start in / is drought, change to 1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_start = np.zeros_like(ds)        # create empty array the same size
    ds_start[start] = 1                 # change indices that are listed in 'start' to 1
    
    # anything with 1 is start of possible drought
    # -5 is definitely NOT a drought
    # and 5 could be a drought but not necessarily
    drought_start = xr.where(ds_start == 1, 1, ds_below_median_precip)
        
    # --- if it's a year that drought should end / not a drought, change to -1
    # --- otherwise, it's a different year (but could still be a drought), is 0
    ds_end = np.zeros_like(ds)          # create empty array the same size
    ds_end[end] = -1                    # change indices that are listed in 'end' to -1
    # combine with precip from before, where definite non-drought year is '-999'
    drought_end = xr.where(ds_end == -1, -999, ds_below_median_precip)
    
    # combine drought start and end arrays.
    # --- definite drought year ==  1
    # --- definitely not a drought == -999
    # --- below mean precip (but may or may not be a drought) == 5
    # --- above mean precip (but may or may not be a drought) == -5
    drought_poss = xr.where(drought_start > 0, drought_start, drought_end)
    
    # --- second pass, get more droughts!
    years = []
    for i in range(len(drought_poss) - 2):
        # check if it is definitely a drought year, followed by a year that has negative precip anom
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == 5):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 1):
            years.append(i+1)
        if drought_poss[i].item() == 1 and (drought_poss[i+1].item() == -5 and drought_poss[i+2].item() == 5):
            years.append(i+1)
            years.append(i+2)
    
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years] = 1
    drought_poss2 = xr.where(ds_years == 1, 1, drought_poss)
    
    # --- repeat process
    years2 = []
    for i in range(len(drought_poss2) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == 5):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 1):
            years2.append(i+1)
        if drought_poss2[i].item() == 1 and (drought_poss2[i+1].item() == -5 and drought_poss2[i+2].item() == 5):
            years2.append(i+1)
            years2.append(i+2)
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years2] = 1
    drought_poss3 = xr.where(ds_years == 1, 1, drought_poss2)
    
    # --- repeat process
    years3 = []
    for i in range(len(drought_poss3) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == 5):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 1):
            years3.append(i+1)
        if drought_poss3[i].item() == 1 and (drought_poss3[i+1].item() == -5 and drought_poss3[i+2].item() == 5):
            years3.append(i+1)
            years3.append(i+2)
    
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years3] = 1
    drought_poss4 = xr.where(ds_years == 1, 1, drought_poss3)
    
    # --- repeat process again (just in case)
    years4 = []
    for i in range(len(drought_poss4) - 2):
        # check if it is a drought year, followed by a null year
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == 5):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 1):
            years4.append(i+1)
        if drought_poss4[i].item() == 1 and (drought_poss4[i+1].item() == -5 and drought_poss4[i+2].item() == 5):
            years4.append(i+1)
            years4.append(i+2)
    
    # combine with overall array (similar to before)
    ds_years = np.zeros_like(ds)
    ds_years[years4] = 1
    drought_poss5 = xr.where(ds_years == 1, 1, drought_poss4)
    
    # --- convert into drought years vs non-drought years
    drought_years = xr.where(drought_poss5 == 1, 1, 0)
    
    return drought_years


def get_drought_years_start_end_thresholds_gufunc(in_array, threshold_start, threshold_end, axis):
    """
    Reshape data and apply the actual 'get_drought_years...' function on a grid-cell basis.
    `threshold` can be:
            - an input array (i.e. a pre-calculated threshold)
            - a single float: in this case, the threshold-array will be calculated across the entire timeseries 
    """
    
    # reshape array to a 2d (lat-lon pair, time)
    data = in_array.reshape(axis, in_array.shape[axis])
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold_start) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        threshold_starts = np.quantile(in_array, threshold_start, axis=axis)   
    else:
        # threshold is already an array
        threshold_starts = threshold_start
        
    # flatten threshold array
    ds_threshold_starts = threshold_starts.flatten()
    
    # Check format for threshold.
    # If it's already a float, convert to quantile first based on the whole input array
    # otherwise, threshold is already an array
    if type(threshold_end) == float:
        # threshold is a float (e.g. 0.2) - need to calculate quantile
        threshold_ends = np.quantile(in_array, threshold_end, axis=axis)   
    else:
        # threshold is already an array
        threshold_ends = threshold_end
        
    # flatten threshold array
    ds_threshold_ends = threshold_ends.flatten()
    
    # Iterate through data and threshold, and apply func
    # NOTE: before, e.g.
    #     results = np.apply_along_axis(get_drought_years_below_threshold, 1, data, threshold)
    # is only able to handle 1 input array (for the axis transformation)
    # but np.apply_along_axis is essentially a wrapper for a for-loop
    
    results =[]
    for i in range(data.shape[0]):
        results.append(get_drought_years_below_start_end_thresholds(data[i,:], ds_threshold_starts[i], ds_threshold_ends[i]))
        
    # reshape to original lat, lon, year shape.
    results_reshaped = np.reshape(results, (in_array.shape[0], in_array.shape[1], in_array.shape[2]))
    return results_reshaped


def get_drought_years_start_end_thresholds_apply(ds_precip, threshold_start, threshold_end):
    # e.g. run on ds.PRECT_mm
    # threshold can either be an 2D-array or a single float
    drought_years = xr.apply_ufunc(get_drought_years_start_end_thresholds_gufunc, ds_precip, threshold_start, threshold_end, input_core_dims=[['year'], [], []],
                          output_dtypes=(float), output_core_dims=[['year']], kwargs={'axis': -1})
    # reorder output so it's consistent with input order
    return drought_years.transpose('year', 'lat', 'lon')

   
    
    