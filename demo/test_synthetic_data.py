__author__ = 'yuafan'

# imports

# function

# script
# input parameter output

# visualization


import matplotlib.pyplot as plt
import numpy as np

time_span = 200
nn_healthy_samples = 18
nn_faulty_samples = 2
nn_observations = 100

time_line = list(range(time_span))
healthy_meta_distr = [[0, 1] for xx in range(time_span)]

faulthy_meta_distr_type_a = [[distr[0]+xx*(2./time_span),distr[1]] for xx, distr in zip(range(time_span), healthy_meta_distr)]
faulthy_meta_distr_type_b = [[distr[0]-xx*(2./time_span),distr[1]] for xx, distr in zip(range(time_span), healthy_meta_distr)]


healthy_sample_pack, faulty_sample_pack = [[] for xx in range(nn_healthy_samples)], [[] for xx in range(nn_faulty_samples)]

for i_nn in range(nn_faulty_samples):

    for i_time in range(time_span):

        healthy_sample_pack[i_nn][i_time].append(np.random.normal(healthy_meta_distr[i_time][0], healthy_meta_distr[i_time][1], nn_observations))

healthy_pack = [np.random.normal(distr[0], distr[1], nn_observations) for distr in healthy_meta_distr]
faulty_pack = [np.random.normal(distr[0], distr[1], nn_observations) for distr in faulthy_meta_distr_type_a]


