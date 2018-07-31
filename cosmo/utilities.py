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
__author__ = 'yuafan'

import numpy as np
import math
# todo: should we replace the math with numpy and not import math here?


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



################################################################################
def compute_p_value(z_scores):
    ''' TODO:
    Uniformity test: compute p-value of the test based on z scores given

    Inputs
    ------
    z_scores: np.array | list of float
    
    Output
    ------
    p-value: float
    
    Usage Example
    -------------
    
    '''
    
    # bundle for calculating p value
    def norm_cdf(x, mu, sigma):
        t = x-mu
        y = 0.5*math.erfc(-t/(sigma*math.sqrt(2.0)))
        if y>1.0:
            y = 1.0
        return y

    def get_arithmetic_p_val(gMu, n):

        if n == 0:
            gP = float('NaN')
            return gP

        amu = 0.5 # mean
        asigma = np.sqrt(1.0/(12.0*n)) # Standard deviation for the mean
        #avar = 1.0/(12.0*n) # Standard deviation for the mean
        #aP = norm.cdf(gMu, amu, s=avar)
        aP = norm_cdf(gMu, amu, asigma)

        return aP

    n_real = np.count_nonzero(~np.isnan(z_scores))
    return get_arithmetic_p_val(float(np.nansum(z_scores))/n_real, np.count_nonzero(~np.isnan(z_scores)) )


################################################################################
def get_p_val(z_scores, period=30):
    ''' TODO:
    Compute p values for a full z-score sequence

    Inputs
    ------
    z_scores: np.array, 1d | list of floats

    Parameters
    ----------
    period: integer, window size for computing uniformity test

    Output
    ------
    p_val: np.array, 1d a sequence of p-values indicate how different a sample is compare to its peers

    Usage Example
    -------------

    '''

    p_val = np.arange(len(z_scores), dtype=np.float)

    for i in range(period):
        x = np.array(z_scores[:i])[~np.isnan(z_scores[:i])]
        p_val[i] = compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan

    for i in range(period, len(z_scores)):
        x = np.array(z_scores[i-period:i])[~np.isnan(z_scores[i-period:i])]
        #print(np.mean(x))
        # print(i, x)
        # print(i, x.size)
        # print(i, np.mean(x))
        p_val[i] = compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan
        #averagedPval[i] = float(-math.log10(stats.t.sf(np.abs(tmp_d5), 3600-1)*2)) if not np.isnan(averagedPval[i]) else np.nan

        # print(p_val[i])

    return p_val
