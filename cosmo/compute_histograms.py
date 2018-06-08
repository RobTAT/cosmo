'''
Copyright (C) Yuantao Fan. All rights reserved.
Author: Yuantao Fan
This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.
You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''

import numpy as np

################################################################################
def compute_derivative(time_series, n_step):
    ''' 
    compute numerical derivative of a uni-variate time series.

    Inputs
    ------
    time_series: np.array, 1d | list, 1d

    Parameters
    ----------
    n_step: integer, step for computing derivatives

    Output
    ------
    list, 1d vector

    Usage Example
    -------------

    TODO: Multivariate version

    '''
    #DerVal = []
    #for elem in range(len(time_series)-n_step):
    #    DerVal.append((time_series[elem+n_step] - time_series[elem])/n_step)
    #print('size of ori', len(_value))
    #print('size of step', len(val_step_scale))
    #return DerVal
    return [(time_series[xx+n_step] - time_series[xx])/n_step for xx in range(len(time_series)-n_step)]

################################################################################
def compute_histogram(value, bin_=[], range_=[], density_ = True):
    '''
    Compute a histogram (density) of given samples.
    
    Inputs
    ------
    value: np.array, 1d | list, 1d
    
    Parameters
    ----------
    bin:     number of bins
    range:   sampling range
    density: whether histogram is normalized, default is True

    Output
    ------
    count:  list, 1d density of value

    Usage Example
    -------------

    '''
    # Normalized_ return a list
    #count, bins, ignored = plt.hist(_val, 60, normed=True, facecolor='blue')  # color... = = really?
    if range_ == [] or bin_== []:
        count, bins = np.histogram(value, density=density_)
    else:
        count, bins = np.histogram(value, bins=bin_, range=range_ , density=density_)
    count = count * (bins[1] - bins[0]) # a must for normalization
    #center = (bins[:-1] + bins[1:]) / 2
    #width = 0.7 * (bins[1] - bins[0])
    return count
