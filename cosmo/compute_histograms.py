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
import Parameters_VACT as pv # TODO: what is this?
import matplotlib.pyplot as plt # TODO: Do you you use this package here?

################################################################################
def compute_derivative(_value, n_step):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------
    
    Parameters
    ----------

    Output
    ------

    Usage Example
    -------------

    '''
    DerVal = []
    for elem in range(len(_value)-n_step):
        DerVal.append((_value[elem+n_step] - _value[elem])/n_step)
    #print('size of ori', len(_value))
    #print('size of step', len(val_step_scale))
    return DerVal

################################################################################
def compute_histogram(value, bin_=pv.Hist_Bins, range_=pv.Hist_BRange):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------
    
    Parameters
    ----------

    Output
    ------

    Usage Example
    -------------

    '''
    # Normalized_ return a list
    #count, bins, ignored = plt.hist(_val, 60, normed=True, facecolor='blue')  # color... = = really?
    count, bins = np.histogram(value, bins=bin_, range=range_ , density=True)
    count = count * (bins[1] - bins[0]) # a must for normalization
    #center = (bins[:-1] + bins[1:]) / 2
    #width = 0.7 * (bins[1] - bins[0])
    return count
