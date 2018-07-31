__author__ = 'yuafan'

################################################################################
######################################################################## imports
################################################################################
import numpy as np

################################################################################
####################################################################### function
################################################################################

################################################################################
######################################################################### script
################################################################################

# todo@yuantao:

# saeedghsh: What I understand is that you want to synthesize two
# lists that you call "healthy and faulty packs".  It would be helpful
# if you specify what the description/specification for these data
# should be. I have changed the code slightly, but not much, mostly
# clean-up and refactoring.  A good way to outline the descriptions
# would be to write test functions.  A test function would take the
# sythesized data and would check if it is constructed correctly, for
# instance if the number of dimentions are correct, if the shape of
# the data matches the input parameters, and if the values of the data
# are within expected interval (if applicable) and with correct type
# (e.g. int, float).  The original code had an indexing error in the
# for-loop (I replaced that, which needs your double-check and
# approval).

# input parameter output
time_span = 200
nn_healthy_samples = 18
nn_faulty_samples = 2
nn_observations = 100

## todo@yuantao: what is a "meta_distr"?
timestamp = list(range(time_span))
healthy_meta_distr = [[0, 1] for _ in timestamp]

## todo@yuantao: what the "types a and b" are supposed to represent? also, type b is never used.
faulty_meta_distr_type_a = [[val[0]+idx*(2./time_span),val[1]] for idx, val in enumerate(healthy_meta_distr)]
faulty_meta_distr_type_b = [[val[0]-idx*(2./time_span),val[1]] for idx, val in enumerate(healthy_meta_distr)]

## todo@yuantao: what is a "sample_pack"?
healthy_sample_pack = [[] for _ in range(nn_healthy_samples)]
faulty_sample_pack = [[] for _ in range(nn_faulty_samples)]

## todo@yuantao:
## the original code (now commented) failed with indexing error.
## can you check if the replacement does the intended operation?
## also note the following change:
## "i_nn in range(nn_faulty_samples)" -->  "i_nn in range(nn_healthy_samples)"
## if this replacement is ok, we won't need the initialization of the
## "healthy_sample_pack" from above

# for i_nn in range(nn_faulty_samples):
#     for i_time in timestamp:
#         healthy_sample_pack[i_nn][i_time].append(
#             np.random.normal(healthy_meta_distr[i_time][0], healthy_meta_distr[i_time][1], nn_observations)
#         )

healthy_sample_pack = [ [np.random.normal(v1,v2, nn_observations)
                         for v1,v2 in healthy_meta_distr ]
                        for i_nn in range(nn_healthy_samples) ]

## todo@yuantao: what is the difference between "sample_pack" and "pack"?
healthy_pack = [np.random.normal(d1,d2, nn_observations) for d1,d2 in healthy_meta_distr]
faulty_pack = [np.random.normal(d1,d2, nn_observations) for d1,d2 in faulty_meta_distr_type_a]

################################################################################
################################################################## visualization
################################################################################
