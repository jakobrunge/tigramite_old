#!/usr/bin/python
# -*- coding: utf-8 -*-
#
# Tigramite -- Time Series Graph based Measures of Information Transfer
#
# Methods are described in:
#    J. Runge, J. Heitzig, V. Petoukhov, and J. Kurths, Phys. Rev. Lett. 108, 258701 (2012)
#    J. Runge, J. Heitzig, N. Marwan, and J. Kurths, Phys. Rev. E 86, 061121 (2012) 
#                                                    AND http://arxiv.org/abs/1210.2748
#    J. Runge, V. Petoukhov, and J. Kurths, Journal of Climate, 27.2 (2014)
#
# Please cite all references when using the method.
#
# Copyright (C) 2012-2014 Jakob Runge <jakobrunge@gmail.com>
# URL: <http://tocsy.pik-potsdam.de/tigramite.php>

"""
Module contains functions for the package tigramite.
"""

#
#  Import essential packages
#
#  Import NumPy for the array object and fast numerics
import numpy
import datetime



def get_index(cdf_name, var, region, period=None, anomalization=None):

    import netCDF4

    cdf = netCDF4.Dataset(cdf_name)
    print cdf.variables.keys()
    if 'latitude' in cdf.variables.keys():
        lat = cdf.variables['latitude'][:]
    else:
        lat = cdf.variables['lat'][:]
    if 'longitude' in cdf.variables.keys():
        lon = cdf.variables['longitude'][:]
    else:
        lon = cdf.variables['lon'][:]

    if lon.min() < 0: 
        lon[ lon < 0] = 360 + lon[ lon < 0]

    latidx = (lat >= region['lat_min']) & (lat <= region['lat_max'])
    lonidx = (lon >= region['lon_min']) & (lon <= region['lon_max'])

    print 'lat grid points in region', lat[latidx], latidx.sum()
    print 'lon grid points in region', lon[lonidx], lonidx.sum()

    if 'level' in cdf.variables.keys():
        lev = cdf.variables['level'][:] == region['level']
        print 'Levels ', cdf.variables['level'][:]
        print 'Chosen level ', cdf.variables['level'][:][lev]
        data = cdf.variables[var][:,lev,latidx,lonidx].squeeze()
    else:
        data = cdf.variables[var][:,latidx,lonidx]


    print data.shape
    ## First zonal mean along longitude
    index = data.mean(axis=2)
    ## Then meridional mean with cos weight
    lat = lat[latidx]
    latr = numpy.deg2rad(lat)
    weights = numpy.cos(latr)
    # print 'weights ', weights
    index = numpy.average(index, axis=1, weights=weights)

    time = cdf.variables['time'][:]
    print time.shape
    if anomalization is not None:
        if 'Daily' in cdf.variables[var].long_name:
            ## Here the leap days are removed making the index shorter!!!
            index, time = anomalize(index, time_cycle=365, anomalize_variance=anomalization['anomalize_variance'],
                              cdf={'remove_leaps':True, 'time':time,
                                       'units':cdf.variables['time'].units,
                                      'anomalization_range':anomalization['anomalization_range'],})
        elif 'Monthly' in cdf.variables[var].long_name:
            index, time = anomalize(index, time_cycle=12, anomalize_variance=anomalization['anomalize_variance'],
                              cdf={'remove_leaps':False, 'time':time,
                                       'units':cdf.variables['time'].units,
                                      'anomalization_range':anomalization['anomalization_range'],})
        else:
            raise ValueError("anomalization only possible for daily or monthly data")

    print time.shape
    if period is not None:
        start_year = period[0]
        end_year = period[1]
        print get_index_from_date(time, cdf_units=cdf.variables['time'].units, year=start_year, month=1, day=1)
        start_year_index = get_index_from_date(time, cdf_units=cdf.variables['time'].units, year=start_year, month=None, day=None)[0]
        end_year_index = get_index_from_date(time, cdf_units=cdf.variables['time'].units, year=end_year, month=None, day=None)[-1]
        cdf.close()

        return index[start_year_index : end_year_index], time[start_year_index : end_year_index]
    else:
        cdf.close()

        return index, time


def time_range( time, zerodate, interval, grid_size, startend, months):
    
    """
    Return a boolean array_mask of the same shape as the input data array with
    only months within list "month" set to true.
    
   
    :type time: array
    :param time: one-dimensional array containing time since zerodate in
                interval specified by "interval".

    :type zerodate: datetime object
    :param zerodate: start date to compute timeaxis.

    :type interval: string
    :param interval: specifies the time step in "time"-array.

    :type grid_size: integer
    :param grid_size: number of processes to construct array for.

    :type startend: tuple
    :param startend: specifies (startyear, endyear), e.g., (1948, 2012) 
                for 2012 included.

    :type months: list
    :param months: contains the selected months, e.g., [1,2,3] for Jan-March.
    
    :rtype: int array
    :returns: the array of shape (len(time), grid_size) with 1 if timeindex is 
                            in "months", 0 else.
    """

    # import datetime for time operations
    import datetime
    from dateutil.relativedelta import relativedelta


    n_time = len(time)

    array_mask = numpy.zeros((n_time, grid_size), dtype='int8')
    year_range = range(startend[0], startend[1]+1)
    take_list = []
    for i in xrange(n_time):
        if interval == 'days':
            date = zerodate + datetime.timedelta(days=int(time[i]))
        elif interval == 'hours':
            date = zerodate + datetime.timedelta(hours=int(time[i]) )
        elif interval == 'months':
            date = zerodate + relativedelta(months=int(time[i]))


#        print date.year, date.month
        if date.year in year_range and date.month in months:
            take_list.append(i)
#            print '\ttaking ', date.year, date.month
    
    array_mask[take_list] = 1

    return array_mask


def common_time_range_netcdf(cdf_list, time_interval = 'months', 
                    start_date=None, end_date=None):
    
    """
    Computes the common time indices of multiple cdf files. Also returns the 
    common time array
    Assumes regular time steps and no missing time values.
   
    :type cdf_list: list of strings
    :param cdf_list: list of paths to cdf files

    :type start_date: tuple
    :param start_date: consider only dates larger or equal to this date

    :type end_date: tuple
    :param end_date: consider only dates smaller or equal to this date


    :type time_interval: string
    :param time_interval: precision up to which the time intersection should 
                        be checked
    
    :rtype: tuple
    :returns: dictionary of intersection indices for every file, array of 
                common time in years, (start date, end date)
    """

    try: import netCDF4
    except: raise Exception("Could not import netCDF4: function 'common_time_range'  doesn't work!")
    # import datetime for time operations
    import datetime, re

    ## Get dates as lists [(year, month, day), ...]
    all_date_list = []
    all_indices_dict = {}
    for cdf_name in cdf_list:
        cdf = netCDF4.Dataset(cdf_name)
        ## This cumbersome line extracts the zerodate from a unit string like "hours since 1800-1-1 00:00:0.0"
        u = re.split(" ", cdf.variables['time'].units)
        interval = u[0]
        assert u[1] == 'since'
        zerodate_raw = re.split("-",u[2])
        zerodate = datetime.datetime(int(zerodate_raw[0]),int(zerodate_raw[1]),int(zerodate_raw[2]),0,0)
#        zerodate = datetime.datetime(int(u[u.find('since')+6:u.find('-')]), int(u[u.find('-')+1]), int(u[u.find('-')+3]),0,0)

        n_time = len(cdf.dimensions['time'])
        time = numpy.copy(cdf.variables['time'][:])
        print cdf_name,
        print ": from unit string '%s' the zerodate %s and interval %s were inferred" %(u, zerodate, interval)
        date_list = []
        for t in xrange(n_time):
            if interval == 'days':
                date = zerodate + datetime.timedelta(days=int(cdf.variables['time'][t]))
            elif interval == 'hours':
                date = zerodate + datetime.timedelta(hours=int(cdf.variables['time'][t]) )

            if time_interval == 'years':
                date_list.append((date.year,))
            elif time_interval == 'months':
                date_list.append((date.year, date.month))
            elif time_interval == 'days':
                date_list.append((date.year, date.month, date.day))
            elif time_interval == 'hours':
                date_list.append((date.year, date.month, date.day, date.hour))

            
        all_date_list.append(date_list)
        ## Map index to date so that the index can be recoverered later
        all_indices_dict[cdf_name] = dict((k,i) for i,k in enumerate(date_list))
        
    # Intersect dates (Note, that the set operator does not preserve the order!)
    intersection = set(all_date_list[0])
    for date_list in all_date_list:
        intersection = intersection & set(date_list)

    ## Restrict to start_date and end_date
    if start_date is None: start_date = min(intersection)
    if end_date is None: end_date = max(intersection)
    intersection = set([d for d in intersection if d>=start_date and d <= end_date])

    ## Get indices of intersection in every cdf file and sort them
    all_intersection_dict = {}
    for cdf_name in cdf_list:
        intersection_indices = [all_indices_dict[cdf_name][x] for x in intersection]
#        print intersection_indices
        intersection_indices.sort()
        all_intersection_dict[cdf_name] = intersection_indices
        
    ## Sort intersection (set operation mixes it up)
    sorted_intersection = list(intersection)
    sorted_intersection.sort()

    ## Create time array in years
    if time_interval == 'years':
        start = sorted_intersection[0][0]
        end = sorted_intersection[-1][0]
    elif time_interval == 'months':
        start = sorted_intersection[0][0] + sorted_intersection[0][1]/12.
        end = sorted_intersection[-1][0] + sorted_intersection[-1][1]/12.
    elif time_interval == 'days':
        start = sorted_intersection[0][0] + sorted_intersection[0][1]/12. + sorted_intersection[0][2]/365.
        end = sorted_intersection[-1][0] + sorted_intersection[-1][1]/12. + sorted_intersection[-1][2]/365.
    elif time_interval == 'hours':
        start = sorted_intersection[0][0] + sorted_intersection[0][1]/12. + sorted_intersection[0][2]/365. + sorted_intersection[0][3]/(365.*24.)
        end = sorted_intersection[-1][0] + sorted_intersection[-1][1]/12. + sorted_intersection[-1][2]/365. + sorted_intersection[-1][3]/(365.*24.)

    time = numpy.linspace(start, end, len(sorted_intersection))

    print "%d common indices found, start %s, end %s" %(len(sorted_intersection), str(sorted_intersection[0]),str( sorted_intersection[-1]))

    return {"indices":all_intersection_dict, "time":time, "startend":(sorted_intersection[0], sorted_intersection[-1])}


def common_time_range(all_time_list):
    
    """
    Computes the common time indices of multiple cdf files. Also returns the common time array
    
   
    :type all_time_list: list of arrays/lists
    :param all_time_list: list of time arrays

    :rtype: tuple
    :returns: dictionary of intersection indices for every time array, array of common time, (start, end)
    """

    n_files = len(all_time_list)

    ## Get dates as lists [(year, month, day), ...]
    all_indices_dict = {}
    for f in range(n_files):
        ## Map index to date so that the index can be recoverered later
        all_indices_dict[f] = dict((k,i) for i,k in enumerate(all_time_list[f]))
        
    # Intersect dates
    intersection = set(all_time_list[0])
    for time_list in all_time_list:
        intersection = intersection & set(time_list)

    ## Get indices of intersection in every cdf file and sort them
    all_intersection_dict = {}
    for f in range(n_files):
        intersection_indices = [all_indices_dict[f][x] for x in intersection]
#        print intersection_indices
        intersection_indices.sort()
        all_intersection_dict[f] = intersection_indices
        
    ## Sort intersection (set operation mixes it up)
    sorted_intersection = list(intersection)
    sorted_intersection.sort()

    ## Create common time array
    time = numpy.linspace(sorted_intersection[0], sorted_intersection[-1], len(sorted_intersection))

    print "%d common indices found, start %s, end %s" %(len(sorted_intersection), str(sorted_intersection[0]),str( sorted_intersection[-1]))

    return {"indices":all_intersection_dict, "time":time, "startend":(sorted_intersection[0], sorted_intersection[-1])}


def get_index_from_date(time, cdf_units, year=None, month=None, day=None):

    import re

    u = cdf_units
    print u
    date_string = re.findall(' [0-9]*-[0-9]*-[0-9]*', u)[0]
    hyphens = [m.start() for m in re.finditer('-', date_string)]
    zerodate = datetime.datetime(int(date_string[:hyphens[0]]),
                                 int(date_string[hyphens[0]+1:hyphens[1]]),
                                 int(date_string[hyphens[1]+1:]))
    interval = u[:u.find(' ')]
    indices = []
    for i in xrange(len(time)):

        if interval == 'days':
            date = zerodate + datetime.timedelta(days=int(time[i]))
        elif interval == 'hours':
            date = zerodate + datetime.timedelta(hours=int(time[i]) )
        if ((year is None or date.year == year) 
        and (month is None or date.month == month) 
        and (day is None or date.day == day)):
            indices.append(i)

    return indices


def remove_leapdays(data, time, cdf_units):
    print 'removing leap days'
    leaplist =  get_index_from_date(time, cdf_units=cdf_units, year=None, month=2, day=29)
    print leaplist
    data = numpy.delete(data,leaplist,axis=0)
    time = numpy.delete(time,leaplist)
    
    return data, time


def anomalize(data, time_cycle, anomalize_variance=False, cdf=None):
    """
    Returns residuals of anomalization.

    Assumes data of shape (T, N) or (T,)

    :type fulldata: array
    :param fulldata: data

    :type time_cycle: integer
    :param time_cycle: anomalization period

    :rtype: array
    :returns: anomalies
    """

    if cdf is None:
        for i in range(int(time_cycle)):
            data[i::int(time_cycle)] -= numpy.mean(data[i::int(time_cycle)], axis = 0)
            if anomalize_variance:
                data[i::int(time_cycle)] /= numpy.std(data[i::int(time_cycle)], axis = 0)
        return data
    else:
        time = cdf['time']
        print time.shape
        if cdf['remove_leaps']:
            data, time = remove_leapdays(data, time, cdf_units=cdf['units'])

        if cdf['anomalization_range'] is not None:
            start_year = cdf['anomalization_range'][0]
            end_year = cdf['anomalization_range'][1]
            start_year_index = get_index_from_date(time, cdf_units=cdf['units'], year=start_year, month=1, day=1)[0]
            end_year_index = get_index_from_date(time, cdf_units=cdf['units'], year=end_year, month=12, day=31)[-1]
            for i in range(time_cycle):
                climatology = numpy.mean(
                    data[ start_year_index+i : end_year_index : time_cycle], axis = 0)
                climatology_std = numpy.std(
                    data[ start_year_index+i : end_year_index : time_cycle], axis = 0)

                data[i::time_cycle] -= climatology
                if anomalize_variance:
                    data[i::time_cycle] /= climatology_std

        else:
            for i in range(int(time_cycle)):
                data[i::time_cycle] -= numpy.mean(data[i::time_cycle], axis = 0)
                if anomalize_variance:
                    data[i::time_cycle] /= numpy.std(data[i::time_cycle], axis = 0)

        return data, time
