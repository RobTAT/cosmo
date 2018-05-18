__author__ = 'yuafan'

import math
from numpy.linalg import norm
import numpy as np

def sorensen(u, v):
    nominator = 0.0
    #Denominator = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        nominator = nominator + abs(v1-v2)
        #Denominator = Denominator + v1 + v2
    return nominator/(sum(u)+sum(v))

def cosine(u, v):
    nom, den_u, den_v = 0.0, 0.0, 0.0
    for vu,vv in zip(np.array(u).flat,np.array(v).flat):
        nom = nom + vu*vv
        den_u = den_u + vu**2
        den_v = den_v + vv**2
    return math.e**(-nom/math.sqrt(den_u*den_v))

def dot(u, v):
    a = 0.0
    a = sum(p*q for p, q in zip(u, v)).tolist()[0]
    return a

def fidelity(u, v):
    a = 0.0
    for i in range(len(u)):
        a += math.sqrt(u[i]*v[i])
    #a = sum(math.sqrt(u*v)).tolist()[0]
    return a

def hellinger(u, v):
    a = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        a += (math.sqrt(v1)-math.sqrt(v2))**2
    return math.sqrt(a/2)

def get_t_statistics(u, v):
    # input u = [size, [mean, var]]
    #print(u,v)
    if (not np.isnan(u[1][0])) and (not np.isnan(v[1][0])):
        if (v[1][1] + u[1][1]) != 0:
            return abs(u[1][0]-v[1][0])/np.sqrt(v[1][1]/v[0] + u[1][1]/u[0])
        else:
            return np.nan
    else:
        return np.nan

def squared_x2(u, v):
    a = 0.0
    for i in range(len(u)):
        a += ((u[i] - v[i])**2)/(u[i]+v[i])
    #a = sum(math.sqrt(u*v)).tolist()[0]
    return a

def hellinger_similarity(u, v):
    a, k = 0.0, 0.0
    for i in range(len(u)):
        a += (math.sqrt(u[i])-math.sqrt(v[i]))**2
    #a = sum(math.sqrt(u*v)).tolist()[0]
    #print(math.sqrt(2*a))
    return math.exp(-(math.sqrt(a/2)))

def euclidean_similarity(u, v):
    dist = norm(u - v)
    return (math.e**(-dist))

def euclidean_l2(u, v):
    return norm(np.array(u) - np.array(v))

def euclidean_Square(u, v):
    return (norm(np.array(u) - np.array(v))**2)/2

def cityblock_similarity(u, v):
    return (1/abs(u - v).sum())

def cityblock_L1(u, v):
    #print(abs(np.array(u) - np.array(v)))
    return (abs(np.array(u) - np.array(v)).sum())

def chebyshev_Li(u, v):
    #return sci_dis.cdist(np.array(u), np.array(v), 'chebyshev')
    val = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        tmp = abs(v1 - v2)
        if tmp > val:
            val = tmp
        else:
            nothing = 0
    return val

def jeffreys(u, v):
    ans = 0.0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        if v1 == 0.:
            ans = ans + 0.
        elif v2 == 0.:
            ans = ans + 1e-20
        else:
            ans = ans + (v1-v2)*math.log(v1/v2)
    return ans

# Earth Mover Distance
def emd_(u, v):
    import emd
    return emd.emd(range(len(u)), range(len(v)), u, v)

# Fast Emd
def femd_(u, v, grounddist_):
    import pyemd
    return pyemd.emd(np.array(u), np.array(v), grounddist_)

def intersection(u, v):
    ans = .0
    for v1,v2 in zip(np.array(u).flat,np.array(v).flat):
        ans = ans + min(v1, v2)
    return ans


