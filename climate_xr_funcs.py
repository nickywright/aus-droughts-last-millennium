# some useful python functions for working in xarray
# specifically used with CESM-LME. Probably would work with other datasets.
# requires: numpy, xarray, cftime

import numpy as np
import xarray as xr
import cftime

# -----------------------------------------------------------------------------
# fix times 
# for CESM-LME files so that we can read them into xarray 
# -----------------------------------------------------------------------------

def fix_times_from_yr0(ds):
    """
    Decode the time variable for xarray using cftime, and output so that it is 'days since 0001'.
    
    This is specifically for files associated with CESM-LME. By default, xarray cannot decode the time
    variable in some files (such as those for CESM-LME), as their time is coded from yr 0 (which is not
    actually a real year)
    
    For CESM-LME, also need to use 'time_bound' (instead of 'time') - otherwise all  data is offset by 1 month.
    """
    
    # fix times using time_bounds: remove a year from the time array and then edit the 'days since XX' 
    # new_times = ds.time - 365  # wrong way!
    time_bound_start = ds.time_bound[:,0] - 365
    
    # convert into human-readable form
    decoded_times = cftime.num2date(time_bound_start, 'days since 0001-01-01 00:00:00', calendar='365_day')
    
    # Some hacks for specific times - not sure why it was adding a day here. If the issue doesn't exist, this won't do anything
    # modify 0850-01-02 to be 0850-01-01 (since everything else starts on the 1st of the month). Keep in same place
    decoded_times2 = [cftime.DatetimeNoLeap(850, 1, 1) if x==cftime.DatetimeNoLeap(850, 1, 2, 1, 0, 0, 6, 6, 2) else x for x in decoded_times]
   
    # modify 1850-01-02 to be 1850-01-01 (since everything else starts on the 1st of the month). Keep in same place
    decoded_times3 = [cftime.DatetimeNoLeap(1850, 1, 1) if x==cftime.DatetimeNoLeap(1850, 1, 2, 0, 59, 59, 999986, 5, 2) else x for x in decoded_times2]
    
    # convert back to numbers
    new_times = cftime.date2num(decoded_times3, 'days since 0001-01-01 00:00:00', calendar='365_day')

    # update the time variable in the dataset
    attrs = {'units': 'days since 0001-01-01 00:00:00', 'calendar': '365_day'}
    dates = xr.Dataset({'time': ('time', new_times, attrs)})
    dates = xr.decode_cf(dates)
    # ds.update({'time':('time', dates['time'], attrs)})
    ds['time'] = dates['time']

def fix_times_from_days_since_yr1(ds):
    """ 
    Function to fix times in files saved from xarray. For POP (ocean) output.
    
    For some reason, exporting monthly data from xarray was resulting in time issues for me.
    This can be avoided by cftime.date2num (e.g. save out as the actual 'days since 0001-01-01'), 
    and then converting back into days since using this function.
    """
    
    new_times = ds.time
    # update the time variable in the dataset
    attrs = {'units': 'days since 0001-01-01 00:00:00', 'calendar': '365_day'}
    dates = xr.Dataset({'time': ('time', new_times, attrs)})
    dates = xr.decode_cf(dates)
    # ds.update({'time':('time', dates['time'], attrs)})
    ds['time'] = dates['time']

def fix_times_from_days_since_850(ds):
    """ 
    Function to fix times in files saved from xarray. For CAM (atmosphere) output.
    
    For some reason, exporting monthly data from xarray was resulting in time issues for me.
    This can be avoided by cftime.date2num (e.g. save out as the actual 'days since 850-01-01'), 
    and then converting back into days since using this function.
    """
    
    new_times = ds.time
    # update the time variable in the dataset
    attrs = {'units': 'days since 0850-01-01 00:00:00', 'calendar': '365_day'}
    dates = xr.Dataset({'time': ('time', new_times, attrs)})
    dates = xr.decode_cf(dates)
    # ds.update({'time':('time', dates['time'], attrs)})
    ds['time'] = dates['time']

# -----------------------------------------------------------------------------
# subset months and/or calculate means
# -----------------------------------------------------------------------------

def juldec_mean(ds):
    """ Calculate the July-December mean for each year """
    
    # subset to only have months between July and December
    ds_july_dec = ds.where((ds['time.month'] >= 7) & (ds['time.month'] <= 12), drop=True)
    ds_july_dec_yr = ds_july_dec.groupby('time.year').mean('time') # get mean per year
    return ds_july_dec_yr

def annual_mean(ds):
    """ Calculate the annual mean for each year """
    ds_yr = ds.groupby('time.year').mean('time')
    return ds_yr

# -----------------------------------------------------------------------------
# monthly to year mean
# -----------------------------------------------------------------------------

# the following is modified from: http://xarray.pydata.org/en/stable/examples/monthly-means.html)
dpm = {'noleap': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '365_day': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'standard': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'proleptic_gregorian': [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       'all_leap': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '366_day': [0, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31],
       '360_day': [0, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30]}

def get_dpm(ds, calendar='365_day'):
    """
    Return an array of days per month corresponding to the months provided in `months`
    This is useful for calculating annual means, taking into account the weight of each month
    """
    month_length = np.zeros(len(ds.time), dtype=int)
    cal_days = dpm[calendar]

    for i, (month, year) in enumerate(zip(ds['time.month'].values, ds['time.year'].values)):
        month_length[i] = cal_days[month]
    return month_length

def weighted_monthly_to_annual_mean(ds):
    """
    Determine annual mean from monthly data, where the length (weight) of each month is incorporated
    """
    month_length = xr.DataArray(get_dpm(ds, calendar='noleap'), coords=[ds.time], name='month_length')
    # Calculate the weights by grouping (can also be done for season)
    weights = month_length.groupby('time.year') / month_length.groupby('time.year').sum()
    # Calculate the weighted average
    ds_weighted = (ds * weights).groupby('time.year').sum(dim='time')
    return ds_weighted
    

# -----------------------------------------------------------------------------
# area averaged mean
# -----------------------------------------------------------------------------

# def average_da(self, dim=None, weights=None):
#     """
#     weighted average for DataArrays
#
#     Parameters
#     ----------
#     dim : str or sequence of str, optional
#         Dimension(s) over which to apply average.
#     weights : DataArray
#         weights to apply. Shape must be broadcastable to shape of self.
#
#     Returns
#     -------
#     reduced : DataArray
#         New DataArray with average applied to its data and the indicated
#         dimension(s) removed.
#
#     """
#
#     if weights is None:
#         return self.mean(dim)
#     else:
#         if not isinstance(weights, xr.DataArray):
#             raise ValueError("weights must be a DataArray")
#
#         # if NaNs are present, we need individual weights
#         if self.notnull().any():
#             total_weights = weights.where(self.notnull()).sum(dim=dim)
#         else:
#             total_weights = weights.sum(dim)
#
#         return (self * weights).sum(dim) / total_weights
#
# def average_ds(self, dim=None, weights=None):
#     """
#     weighted average for Datasets
#
#     Parameters
#     ----------
#     dim : str or sequence of str, optional
#         Dimension(s) over which to apply average.
#     weights : DataArray
#         weights to apply. Shape must be broadcastable to shape of data.
#
#     Returns
#     -------
#     reduced : Dataset
#         New Dataset with average applied to its data and the indicated
#         dimension(s) removed.
#
#     """
#
#     if weights is None:
#         return self.mean(dim)
#     else:
#         return self.apply(average_da, dim=dim, weights=weights)
#
# def xarray_average(data, dim=None, weights=None):
#     """
#     weighted average for xray objects
#
#     Parameters
#     ----------
#     data : Dataset or DataArray
#         the xray object to average over
#     dim : str or sequence of str, optional
#         Dimension(s) over which to apply average.
#     weights : DataArray
#         weights to apply. Shape must be broadcastable to shape of data.
#
#     Returns
#     -------
#     reduced : Dataset or DataArray
#         New xray object with average applied to its data and the indicated
#         dimension(s) removed.
#
#     """
#
#     if isinstance(data, xr.Dataset):
#         return average_ds(data, dim, weights)
#     elif isinstance(data, xr.DataArray):
#         return average_da(data, dim, weights)
#     else:
#         raise ValueError("date must be an xray Dataset or DataArray")
#
# def weighted_mean_CAM(ds):
#     """
#     Calculate weighted-area mean for CAM (atmosphere) files and keep files in xarray
#     Uses xarray_average, and lat/lon are hard-coded here to correspond to CAM files
#
#     """
#     lats_radians = np.deg2rad(ds.lat)      # convert latitude to radians
#     # use the cosine of the latitudes (in radians) as weights for the average
#     lats_weights = np.cos(lats_radians)
#
#     # find the zonal mean by averaging along latitude circles
#     variable_ave_zonal = ds.mean(dim='lon', keep_attrs=True)
#     # take the weighted average of those using weights calculated earlier
#
#     # variable_weighted_ave = np.average(variable_ave_zonal, axis=1, weights=lats_weights) # returns numpy array rather than xarray
#
#     # do the weighted average but keep it in xarray (so that we keep time). gives the same answer as above
#     variable_weighted_ave_xr = xarray_average(variable_ave_zonal, dim='lat', weights=lats_weights)
#     return variable_weighted_ave_xr


def weighted_mean_CAM(ds):
    # redone, based on: http://xarray.pydata.org/en/stable/examples/area_weighted_temperature.html
    lats_radians = np.deg2rad(ds.lat)      # convert latitude to radians
    # use the cosine of the latitudes (in radians) as weights for the average
    lats_weights = np.cos(lats_radians)
    
    ds_weighted = ds.weighted(lats_weights)
    ds_weighted_mean = ds_weighted.mean(("lon", "lat"))
    return ds_weighted_mean
    
def weighted_mean_POP(ds):
    """ Calculte weighted mean for POP. This requires TAREA, and weights based on the area of each gridcell
    This has not been tested.
    """
    ds_mean = (ds * ds.TAREA).sum(dim=['nlat', 'nlon']) / ds.TAREA.sum()
    return ds_mean


# -----------------------------------------------------------------------------
# import files
# -----------------------------------------------------------------------------

def import_full_forcing_variable_cam(filepath, ensemble_number, variable):
    """
    Import full forcing CAM (atmosphere) files using xarray.
    Here we specify the filepath, ensemble_number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in.
    
    THIS replaces a previous function named 'import_fullforcing_cam'
    
    edits: 14/Oct/19 - made generic for all variable types, automated removing coords/datavars.
                     - also changed so for ensemble member #1, it checks to see if the 850-1849 file exists. 
                       If the file doesn't exist, it imports the 1700-1849 file instead
            12/08/22 - changed the way time is updated, since xarray complains now
    """
    print('...importing %s from ensemble member %s' % (variable, ensemble_number))
    if ensemble_number == '001':
        # check to see if the file from 850 CE exists. If it does, import
        try:
            # try and import 850-1849 CE files
            ds_p1 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.08*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
            print('...imported %s from 850 CE' % variable)
        except IOError:
            ds_p1 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.17*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
            print('...imported %s from 1700 CE' % variable)
        ds_p2 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.18*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
        
        # fix times for part 1
        new_days_since_p1 = ds_p1.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p1.update({'time':('time', dates['time'], attrs)})
        ds_p1['time'] = dates['time']

        # fix times for part 2
        new_days_since_p2 = ds_p2.time_bnds.values[:,0] + 15 + (365*1000)  # use time bounds instead. add 15 to get middle of month. change to be since 850
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p2, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2['time'] = dates['time']
        ds_cam = xr.concat([ds_p1, ds_p2], dim='time')

    elif ensemble_number == '011':
        # Specific for ensemble member 11, as time is encoded differently from the other members for the 1850-2005 portion.
        ds_p1 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.0*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
        ds_p2 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.1*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')

        # fix times for part 1
        new_days_since_p1 = ds_p1.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p1.update({'time':('time', dates['time'], attrs)})
        ds_p1['time'] = dates['time']
        # fix times for part 2
        new_days_since_p2 = ds_p2.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month. change to be since 850
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p2, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2['time'] = dates['time']
        
        ds_cam = xr.concat([ds_p1, ds_p2], dim='time')
        
    else:
        # Import any other ensemble member
        ds_p1 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.0*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
        ds_p2 = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number + '.cam.h*.' + variable + '.1*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')

        # fix times for part 1
        new_days_since_p1 = ds_p1.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p1.update({'time':('time', dates['time'], attrs)})
        ds_p1['time'] = dates['time']

        # fix times for part 2
        new_days_since_p2 = ds_p2.time_bnds.values[:,0] + 15 + (365*1000)  # use time bounds instead. add 15 to get middle of month. change to be since 850
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p2, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2['time'] = dates['time']
        ds_cam = xr.concat([ds_p1, ds_p2], dim='time')
    
    # just in case - re-do time variable so it's all in the same format
    # (sometimes, part would be as datetime and some as cftime...)
    time_num = ds_cam.time.values
    new_time3 = cftime.date2num(time_num, 'days since 850-01-01 00:00:00',  calendar='365_day')
    dates = xr.Dataset({'time': ('time', new_time3, attrs)})
    dates = xr.decode_cf(dates)
    # ds_cam.update({'time':('time', dates['time'], attrs)})
    ds_cam['time'] = dates['time']

    # get rid of unneeded variables
    datavars = ds_cam.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        else: datavars_to_remove.append(i)
    ds_cam = ds_cam.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds_cam.coords
    coords_to_remove = []
    for i in coords:
        if i == 'lat': pass
        elif i == 'lon': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds_cam = ds_cam.drop(coords_to_remove)
    
    return ds_cam


def import_full_forcing_precip(filepath, case_no):
    """
    Function to import CESM-LME full-forcing precipitation files 
    and convert precip units into from m/s to mm/month.
    Conversion for m/s to mm/month is: value x 1000 [m to mm] x (60 x 60 x 24 x month-length) [i.e. for min, hour, day, month]
    
    Inputs: filepath to wherever files are, ensemble number to import
    Outputs: xr.Dataset with the variables 'PRECT' (original, in m/s) and 'PRECT_mm' (converted into mm/month)
    
    """
    ff_precip = import_full_forcing_variable_cam(filepath, case_no, 'PRECT')
    # convert precip from m/s into mm/month
    month_length = xr.DataArray(get_dpm(ff_precip, calendar='noleap'), 
                                coords=[ff_precip.time], name='month_length')
    ff_precip['PRECT_mm'] = ff_precip.PRECT * 1000 * 60 * 60 * 24 * month_length
    
    # add precip attributes back but with our converted units.
    ff_precip.PRECT_mm.attrs = ff_precip.PRECT.attrs   # add normal attributes back
    ff_precip.PRECT_mm.attrs['units'] = 'mm/month'
    
    return ff_precip

def import_control_variable_cam(filepath, variable):
    """
    Import control files for CAM (atmosphere) files using xarray.
    Here we specify the filepath, ensemble_number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in.
    
    """
    
    if variable == 'PRECT':
        # if PRECT, actually import PRECC and PRECL
        ds_precc = xr.open_mfdataset(filepath + 'b.e11.B1850C5CN.f19_g16.0850cntl.001.cam.h0.PRECC.*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
        ds_precl = xr.open_mfdataset(filepath + 'b.e11.B1850C5CN.f19_g16.0850cntl.001.cam.h0.PRECL.*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')

        # fix times for part 1
        new_days_since_p1 = ds_precc.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 651-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        ds_precc['time'] = dates['time']
        
        # fix times for part 1
        new_days_since_p1 = ds_precl.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 651-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_precl.update({'time':('time', dates['time'], attrs)})
        ds_precl['time'] = dates['time']

        ds_precc['PRECT'] = ds_precc.PRECC + ds_precl.PRECL
        ds_cam = ds_precc
        # # just in case - re-do time variable so it's all in the same format
        # # (sometimes, part would be as datetime and some as cftime...)
        # time_num = ds_cam.time.values
        # new_time3 = cftime.date2num(time_num, 'days since 850-01-01 00:00:00',  calendar='365_day')
        # dates = xr.Dataset({'time': ('time', new_time3, attrs)})
        # dates = xr.decode_cf(dates)
        # ds_cam.update({'time':('time', dates['time'], attrs)})
    
        # get rid of unneeded variables
        datavars = ds_cam.data_vars
        datavars_to_remove = []
        for i in datavars:
            if i == 'PRECT': pass
            else: datavars_to_remove.append(i)
        ds_cam = ds_cam.drop(datavars_to_remove)

        # get rid of unneeded coordinates
        coords = ds_cam.coords
        coords_to_remove = []
        for i in coords:
            if i == 'lat': pass
            elif i == 'lon': pass
            elif i == 'time': pass
            else: coords_to_remove.append(i)
        ds_cam = ds_cam.drop(coords_to_remove)
        
        # convert precip from m/s into mm/month
        month_length = xr.DataArray(get_dpm(ds_cam, calendar='noleap'), 
                                    coords=[ds_cam.time], name='month_length')
        ds_cam['PRECT_mm'] = ds_cam.PRECT * 1000 * 60 * 60 * 24 * month_length
        
    else:
        ds_cam = xr.open_mfdataset(filepath + 'b.e11.B1850C5CN.f19_g16.0850cntl.001.cam.h0.' + variable + '.*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')

        # fix times for part 1
        new_days_since_p1 = ds_cam.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_cam.update({'time':('time', dates['time'], attrs)})
        ds_cam['time'] = dates['time']
        
        # # just in case - re-do time variable so it's all in the same format
        # # (sometimes, part would be as datetime and some as cftime...)
        # time_num = ds_cam.time.values
        # new_time3 = cftime.date2num(time_num, 'days since 850-01-01 00:00:00',  calendar='365_day')
        # dates = xr.Dataset({'time': ('time', new_time3, attrs)})
        # dates = xr.decode_cf(dates)
        # ds_cam.update({'time':('time', dates['time'], attrs)})
        
        # get rid of unneeded variables
        datavars = ds_cam.data_vars
        datavars_to_remove = []
        for i in datavars:
            if i == variable: pass
            else: datavars_to_remove.append(i)
        ds_cam = ds_cam.drop(datavars_to_remove)
    
        # get rid of unneeded coordinates
        coords = ds_cam.coords
        coords_to_remove = []
        for i in coords:
            if i == 'lat': pass
            elif i == 'lon': pass
            elif i == 'time': pass
            else: coords_to_remove.append(i)
        ds_cam = ds_cam.drop(coords_to_remove)
        
    return ds_cam

def import_full_forcing_variable_pop(filepath, ensemble_number, variable):
    """ 
    Import full forcing POP (ocean) files using xarray. 
    Here we specify the filepath, ensemble number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in. 
    
    The ensemble member should be in a 3-digit format

    NOTE: this has not yet been tested
    """
    ds = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + ensemble_number 
                               + '.pop.h.*.nc', decode_times=False, chunks={'time': 10}, combine='by_coords')

    if variable == 'D20':
        # clean up ds_D20
        ds = ds.set_coords(['TLAT', 'TLONG', 'ULAT', 'ULONG'])
        # rename into lower case
        ds = ds.rename({'TIME': 'time', 'TIME_bnds': 'time_bound', 'NLAT': 'nlat', 'NLON': 'nlon', 'bnds':'d2'})
        # get rid of time dimension for lats/lons - not sure how it got there
        ds['TLAT'] = ds['TLAT'].isel(time=0)
        ds['TLONG'] = ds['TLONG'].isel(time=0)
        ds['ULAT'] = ds['ULAT'].isel(time=0)
        ds['ULONG'] = ds['ULONG'].isel(time=0)
        fix_times_from_yr0(ds)
        ds = ds.drop(('time_bound'))
        del ds['D20'].attrs['coordinates']  # del this attribute, otherwise it will cause a problem later
        ds['D20'] = ds.D20.astype('float32')  # convert from float64 to float 32
    else:
        fix_times_from_yr0(ds)
    
    # get rid of unneeded variables
    datavars = ds.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        elif i == 'TAREA': pass
        else: datavars_to_remove.append(i)
    ds = ds.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds.coords
    coords_to_remove = []
    for i in coords:
        if i == 'nlat': pass
        elif i == 'nlon': pass
        elif i == 'TLAT': pass
        elif i == 'TLONG': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds = ds.drop(coords_to_remove)
    
    # set TAREA as a coordinate
    ds['TAREA'] = ds['TAREA'].isel(time=0)
    ds = ds.set_coords(['TAREA'])
    
    # get rid of z_t
    if variable == 'SST':
        ds = ds.sel(z_t=0)
        
    return ds

def import_control_variable_pop(filepath, variable):
    """
    Import control files for POP (ocean) files using xarray.
    Here we specify the filepath, ensemble_number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in.
    
    """
    ds = xr.open_mfdataset(filepath + 'b.e11.B1850C5CN.f19_g16.0850cntl.001.pop.*' + variable + '.*.nc', decode_times=False, chunks={'time': 100}, combine='by_coords')
    
    fix_times_from_yr0(ds)
    
    
    # get rid of unneeded variables
    datavars = ds.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        elif i == 'TAREA': pass
        else: datavars_to_remove.append(i)
    ds = ds.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds.coords
    coords_to_remove = []
    for i in coords:
        if i == 'nlat': pass
        elif i == 'nlon': pass
        elif i == 'TLAT': pass
        elif i == 'TLONG': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds = ds.drop(coords_to_remove)
    
    # set TAREA as a coordinate
    ds['TAREA'] = ds['TAREA'].isel(time=0)
    ds = ds.set_coords(['TAREA'])
    
    # get rid of z_t
    if variable == 'SST':
        ds = ds.sel(z_t=ds.z_t.values[0])
    
    return ds


def import_single_forcing_variable_cam(filepath, forcing_type, ensemble_number, variable):
    """
    Import single forcing CAM (atmosphere) files using xarray.
    Here we specify the filepath, ensemble_number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in.

    """
    print('...importing %s from %s ensemble member %s' % (variable, forcing_type, ensemble_number))
    if forcing_type == 'OZONE_AER':
        # ozone only exists from 1850 onwards
        filename = '%s/b.e11.BLMTRC5CN.f19_g16.%s.%s.cam.h0.%s.18*.nc' % (filepath, forcing_type, ensemble_number, variable)
        ds_cam = xr.open_mfdataset(filename, decode_times=False, chunks={'time': 100}, combine='by_coords')

        new_days_since_p2 = ds_cam.time_bnds.values[:,0] + 15 + (365*1000) # use time bounds instead. add 15 to get middle of month. change to be since 850
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p2, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_cam['time'] = dates['time']

    else:
        # Import any other ensemble member
        filename_p1 = '%s/b.e11.BLMTRC5CN.f19_g16.%s.%s.cam.h0.%s.08*.nc' % (filepath, forcing_type, ensemble_number, variable)
        filename_p2 = '%s/b.e11.BLMTRC5CN.f19_g16.%s.%s.cam.h0.%s.1*.nc' % (filepath, forcing_type, ensemble_number, variable)
        ds_p1 = xr.open_mfdataset(filename_p1, decode_times=False, chunks={'time': 100}, combine='by_coords')
        ds_p2 = xr.open_mfdataset(filename_p2, decode_times=False, chunks={'time': 100}, combine='by_coords')

        # fix times for part 1
        new_days_since_p1 = ds_p1.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p1, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p1.update({'time':('time', dates['time'], attrs)})
        ds_p1['time'] = dates['time']

        # fix times for part 2
        new_days_since_p2 = ds_p2.time_bnds.values[:,0] + 15 # use time bounds instead. add 15 to get middle of month. change to be since 850
        attrs = {'units': 'days since 850-01-01 00:00:00', 'calendar': '365_day'}
        dates = xr.Dataset({'time': ('time', new_days_since_p2, attrs)})
        dates = xr.decode_cf(dates)
        # ds_p2.update({'time':('time', dates['time'], attrs)})
        ds_p2['time'] = dates['time']
        ds_cam = xr.concat([ds_p1, ds_p2], dim='time')
    
    # just in case - re-do time variable so it's all in the same format
    # (sometimes, part would be as datetime and some as cftime...)
    time_num = ds_cam.time.values
    new_time3 = cftime.date2num(time_num, 'days since 850-01-01 00:00:00',  calendar='365_day')
    dates = xr.Dataset({'time': ('time', new_time3, attrs)})
    dates = xr.decode_cf(dates)
    # ds_cam.update({'time':('time', dates['time'], attrs)})
    ds_cam['time'] = dates['time']

    # get rid of unneeded variables
    datavars = ds_cam.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        else: datavars_to_remove.append(i)
    ds_cam = ds_cam.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds_cam.coords
    coords_to_remove = []
    for i in coords:
        if i == 'lat': pass
        elif i == 'lon': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds_cam = ds_cam.drop(coords_to_remove)
    
    return ds_cam
    
def import_single_forcing_variable_pop(filepath, forcing_type, ensemble_number, variable):
    """ 
    Import single forcing POP (ocean) files using xarray. 
    Here we specify the filepath, ensemble number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in. 
    
    The ensemble member should be in a 3-digit format
    NOTE: this has not yet been tested
    """
    ds = xr.open_mfdataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.' + forcing_type + '.' + ensemble_number 
                               + '.pop.h.*.nc', decode_times=False, chunks={'time': 10}, combine='by_coords')

    if variable == 'D20':
        # clean up ds_D20
        ds = ds.set_coords(['TLAT', 'TLONG', 'ULAT', 'ULONG'])
        # rename into lower case
        ds = ds.rename({'TIME': 'time', 'TIME_bnds': 'time_bound', 'NLAT': 'nlat', 'NLON': 'nlon', 'bnds':'d2'})
        # get rid of time dimension for lats/lons - not sure how it got there
        ds['TLAT'] = ds['TLAT'].isel(time=0)
        ds['TLONG'] = ds['TLONG'].isel(time=0)
        ds['ULAT'] = ds['ULAT'].isel(time=0)
        ds['ULONG'] = ds['ULONG'].isel(time=0)
        fix_times_from_yr0(ds)
        ds = ds.drop(('time_bound'))
        del ds['D20'].attrs['coordinates']  # del this attribute, otherwise it will cause a problem later
        ds['D20'] = ds.D20.astype('float32')  # convert from float64 to float 32
    else:
        fix_times_from_yr0(ds)
    
    # get rid of unneeded variables
    datavars = ds.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        elif i == 'TAREA': pass
        else: datavars_to_remove.append(i)
    ds = ds.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds.coords
    coords_to_remove = []
    for i in coords:
        if i == 'nlat': pass
        elif i == 'nlon': pass
        elif i == 'TLAT': pass
        elif i == 'TLONG': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds = ds.drop(coords_to_remove)
    
    # set TAREA as a coordinate
    ds['TAREA'] = ds['TAREA'].isel(time=0)
    ds = ds.set_coords(['TAREA'])
    
    # get rid of z_t
    if variable == 'SST':
        ds = ds.sel(z_t=0)
        
    return ds

def import_single_forcing_variable_pop_ozone(filepath, ensemble_number, variable):
    """ 
    Import single forcing POP (ocean) files using xarray. 
    Here we specify the filepath, ensemble number of interest, and the variable.
    The function will remove all other data variables and coordinates except the variable we're interested in. 
    
    The ensemble member should be in a 3-digit format
    NOTE: this has not yet been tested
    """
    ds = xr.open_dataset(filepath + 'b.e11.BLMTRC5CN.f19_g16.OZONE_AER.' + ensemble_number 
                               + '.pop.h.' + variable + '.185001-200512.nc', decode_times=False, chunks={'time': 10})

    if variable == 'D20':
        # clean up ds_D20
        ds = ds.set_coords(['TLAT', 'TLONG', 'ULAT', 'ULONG'])
        # rename into lower case
        ds = ds.rename({'TIME': 'time', 'TIME_bnds': 'time_bound', 'NLAT': 'nlat', 'NLON': 'nlon', 'bnds':'d2'})
        # get rid of time dimension for lats/lons - not sure how it got there
        ds['TLAT'] = ds['TLAT'].isel(time=0)
        ds['TLONG'] = ds['TLONG'].isel(time=0)
        ds['ULAT'] = ds['ULAT'].isel(time=0)
        ds['ULONG'] = ds['ULONG'].isel(time=0)
        fix_times_from_yr0(ds)
        ds = ds.drop(('time_bound'))
        del ds['D20'].attrs['coordinates']  # del this attribute, otherwise it will cause a problem later
        ds['D20'] = ds.D20.astype('float32')  # convert from float64 to float 32
    else:
        fix_times_from_yr0(ds)
    
    # get rid of unneeded variables
    datavars = ds.data_vars
    datavars_to_remove = []
    for i in datavars:
        if i == variable: pass
        elif i == 'TAREA': pass
        else: datavars_to_remove.append(i)
    ds = ds.drop(datavars_to_remove)

    # get rid of unneeded coordinates
    coords = ds.coords
    coords_to_remove = []
    for i in coords:
        if i == 'nlat': pass
        elif i == 'nlon': pass
        elif i == 'TLAT': pass
        elif i == 'TLONG': pass
        elif i == 'time': pass
        else: coords_to_remove.append(i)
    ds = ds.drop(coords_to_remove)
    
    # set TAREA as a coordinate
    # ds['TAREA'] = ds['TAREA'].isel(time=0)
    # ds = ds.set_coords(['TAREA'])
    
    # get rid of z_t
    if variable == 'SST':
        ds = ds.sel(z_t=0)
        
    return ds


# -----------------------------------------------------------------------------
# plotting helpers
# -----------------------------------------------------------------------------
 
def fix_lon_lat_pcolormesh(df):
    """fix plotting issue (for specifically CESM-LME),
    where by default, pcolormesh will be offset by a grid cell.
    Based on: https://bairdlangenbrunner.github.io/python-for-climate-scientists/matplotlib/pcolormesh-grid-fix.html
    EDIT: as of matplotlib 3.3, this is no longer needed (but good to check between shading='flat' and shading='auto' when plotting in pcolormesh)""" 
    
    # extend longitude by 2
    lon_extend = np.zeros(df.lon.size+2)
    # fill in internal values
    lon_extend[1:-1] = df.lon # fill up with original values
    # fill in extra endpoints
    lon_extend[0] = df.lon[0]-np.diff(df.lon)[0]
    lon_extend[-1] = df.lon[-1]+np.diff(df.lon)[-1]
    # calculate the midpoints
    lon_pcolormesh_midpoints = lon_extend[:-1]+0.5*(np.diff(lon_extend))
    
    # extend latitude by 2
    lat_extend = np.zeros(df.lat.size+2)
    # fill in internal values
    lat_extend[1:-1] = df.lat
    # fill in extra endpoints
    lat_extend[0] = df.lat[0]-np.diff(df.lat)[0]
    lat_extend[-1] = df.lat[-1]+np.diff(df.lat)[-1]
    # calculate the midpoints
    lat_pcolormesh_midpoints = lat_extend[:-1]+0.5*(np.diff(lat_extend))
    
    return lon_pcolormesh_midpoints, lat_pcolormesh_midpoints


