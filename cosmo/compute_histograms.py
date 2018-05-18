import numpy as np
import Parameters_VACT as pv
import matplotlib.pyplot as plt

def compute_derivative(_value, n_step):
    DerVal = []
    for elem in range(len(_value)-n_step):
        DerVal.append((_value[elem+n_step] - _value[elem])/n_step)
    #print('size of ori', len(_value))
    #print('size of step', len(val_step_scale))
    return DerVal

def compute_histogram(value, bin_=pv.Hist_Bins, range_=pv.Hist_BRange):
    # Normalized_ return a list
    #count, bins, ignored = plt.hist(_val, 60, normed=True, facecolor='blue')  # color... = = really?
    count, bins = np.histogram(value, bins=bin_, range=range_ , density=True)
    count = count * (bins[1] - bins[0]) # a must for normalization
    #center = (bins[:-1] + bins[1:]) / 2
    #width = 0.7 * (bins[1] - bins[0])
    return count
