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
# __author__ = 'yuafan'

from __future__ import print_function, division

import math
import numpy.linalg
import numpy as np

################################################################################
def sorensen(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    nominator = 0.0
    #Denominator = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        nominator = nominator + abs(v1-v2)
        #Denominator = Denominator + v1 + v2
    return nominator/(sum(u)+sum(v))

################################################################################
def cosine(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------
    
    Output
    ------

    Usage Example
    -------------

    '''
    nom, den_u, den_v = 0.0, 0.0, 0.0
    for vu,vv in zip(np.array(u).flat,np.array(v).flat):
        nom = nom + vu*vv
        den_u = den_u + vu**2
        den_v = den_v + vv**2
    return math.e**(-nom/math.sqrt(den_u*den_v))

################################################################################
def dot(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------
    
    Output
    ------

    Usage Example
    -------------

    '''
    a = 0.0
    a = sum(p*q for p, q in zip(u, v)).tolist()[0]
    return a

################################################################################
def fidelity(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    a = 0.0
    for i in range(len(u)):
        a += math.sqrt(u[i]*v[i])
    #a = sum(math.sqrt(u*v)).tolist()[0]
    return a

################################################################################
def hellinger(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    a = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        a += (math.sqrt(v1)-math.sqrt(v2))**2
    return math.sqrt(a/2)

################################################################################
def get_t_statistics(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    # input u = [size, [mean, var]]
    #print(u,v)
    if (not np.isnan(u[1][0])) and (not np.isnan(v[1][0])):
        if (v[1][1] + u[1][1]) != 0:
            return abs(u[1][0]-v[1][0])/np.sqrt(v[1][1]/v[0] + u[1][1]/u[0])
        else:
            return np.nan
    else:
        return np.nan

################################################################################
def squared_x2(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    a = 0.0
    for i in range(len(u)):
        a += ((u[i] - v[i])**2)/(u[i]+v[i])
    #a = sum(math.sqrt(u*v)).tolist()[0]
    return a

################################################################################
def hellinger_similarity(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    a, k = 0.0, 0.0
    for i in range(len(u)):
        a += (math.sqrt(u[i])-math.sqrt(v[i]))**2
    #a = sum(math.sqrt(u*v)).tolist()[0]
    #print(math.sqrt(2*a))
    return math.exp(-(math.sqrt(a/2)))

################################################################################
def euclidean_similarity(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    dist = numpy.linalg.norm(u - v)
    return (math.e**(-dist))

################################################################################
def euclidean_l2(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    return numpy.linalg.norm(np.array(u) - np.array(v))

################################################################################
def euclidean_Square(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    return (numpy.linalg.norm(np.array(u) - np.array(v))**2)/2

################################################################################
def cityblock_similarity(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    return (1/abs(u - v).sum())

################################################################################
def cityblock_L1(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    #print(abs(np.array(u) - np.array(v)))
    return (abs(np.array(u) - np.array(v)).sum())

################################################################################
def chebyshev_Li(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    #return sci_dis.cdist(np.array(u), np.array(v), 'chebyshev')
    val = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        tmp = abs(v1 - v2)
        if tmp > val:
            val = tmp
        else:
            nothing = 0
    return val

################################################################################
def jeffreys(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    ans = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        if v1 == 0.:
            ans = ans + 0.
        elif v2 == 0.:
            ans = ans + 1e-20
        else:
            ans = ans + (v1-v2)*math.log(v1/v2)
    return ans

################################################################################
# Earth Mover Distance
def emd_(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    import emd
    return emd.emd(range(len(u)), range(len(v)), u, v)

################################################################################
# Fast Emd
def femd_(u, v, grounddist_):
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
    import pyemd
    return pyemd.emd(np.array(u), np.array(v), grounddist_)

################################################################################
def intersection(u, v):
    ''' TODO:
    <short description of the method>
    
    Inputs
    ------

    Output
    ------

    Usage Example
    -------------

    '''
    ans = .0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        ans = ans + min(v1, v2)
    return ans


