__author__ = 'yuafan'

import cosmo_v0.metrics as metrics

import COSMO.Parameters_VACT as pv
import COSMO.Metrics as cm_
import numpy as np
import math
import copy


class Anomaly(object):

    def __init__(self, n_units, n_variations):
        self.n_units = n_units
        self.n_variations = n_variations

        #self.models_samples = []
        #self.models_samples = [[[] for var in range(n_variations)] for unit in range(n_units)]
        self.models_samples = [[[] for unit in range(n_units)] for var in range(n_variations)]

        self.distance = None
        self.distance_matrix = None

    def set_samples(self, samples):
        self.models_samples = samples

    def set_distance(self, distance='Hellinger'):
        if distance == 'Hellinger':
            self.distance = metrics.hellinger
        if distance == 'EMD':
            self.distance = metrics.emd_
        if distance == 'Euclidean':
            self.distance = metrics.euclidean_l2
        if distance == 'Cityblock':
            self.distance = metrics.cityblock_L1
        if distance == 'Cosine':
            self.distance = metrics.cosine

    def compute_distance_matrix(self):

        def symmetrize(a):
            return a + a.T - np.diag(a.diagonal())

        n_samples = self.n_units * self.n_variations
        distance_matrix = np.zeros((n_samples, n_samples))

        for irow in range(n_samples):
            for icol in range(irow, n_samples):
                #print(n_samples)
                #print(irow, icol)
                #print(irow%self.n_variations, irow/self.n_variations)
                #print(icol%self.n_variations, icol/self.n_variations)
                #model_a = self.models_samples[irow%self.n_variations][irow/self.n_variations]
                model_a = self.models_samples[int(irow%self.n_variations)][int(irow/self.n_variations)]
                #print('len_histogram', len(model_a))
                #print('type_histogram', type(model_a))
                model_b = self.models_samples[int(icol%self.n_variations)][int(icol/self.n_variations)]
                if model_a != [] and model_b != []:
                    distance_matrix[irow][icol] = self.distance(model_a, model_b)
                else:
                    distance_matrix[irow][icol] = np.nan
        #return symmetrize(distance_matrix)
        self.distance_matrix = symmetrize(distance_matrix)

    def get_z_score(self):

        n_var = self.n_variations

        z_score_samples = np.zeros(self.n_units)
        row_sum = np.zeros(self.distance_matrix.shape[1])
        #print('row_sum', row_sum)

        for i_row in range(self.distance_matrix.shape[1]):
            #print(self.distance_matrix[i_row])
            if np.all(np.isnan(self.distance_matrix[i_row]) == True):
                row_sum[i_row] = np.inf
            elif not self.distance_matrix[i_row].any():
                row_sum[i_row] = np.inf
            else:
                row_sum[i_row] = np.nansum(self.distance_matrix[i_row])
                if row_sum[i_row] == 0.:
                    row_sum[i_row] = np.inf

        #most_central_unit
        #print('min_rowsum', min(row_sum))
        #print(row_sum)
        #mcp_index = np.where(row_sum==min(row_sum))
        mcp_index = np.argmin(row_sum)
        mcp_distribution = self.distance_matrix[mcp_index]

        #mcu, mcv = mcp_index/n_var, mcp_index%n_var
        #most_central_variation of the central unit
        #mcv = np.where(self.distance_matrix[mcu][mcu/n_var*n_var:mcu/n_var*n_var+n_var]==min(self.distance_matrix[mcu][mcu/n_var*n_var:mcu/n_var*n_var+n_var]))

        for i_unit in range(self.n_units):
            '''
            print(len(self.distance_matrix))
            print(len(self.distance_matrix[0]))
            print(self.distance_matrix)
            print(self.distance_matrix[0])
            print(self.distance_matrix[1][0])
            print(self.distance_matrix[0][i_unit*n_var+n_var-1])
            print('mcp_', mcp_index, i_unit*n_var+n_var-1)
            '''
            distance_unit2mcp = self.distance_matrix[mcp_index][i_unit*n_var+n_var-1]
            #print('distance_unit2mcp', distance_unit2mcp)
            if np.isnan(distance_unit2mcp):
                z_score_samples[i_unit] = np.nan
            else:
                z_score_samples[i_unit] = 1.*len([d for d in mcp_distribution if d >= distance_unit2mcp] ) / len(mcp_distribution)
                #print(z_score_samples[i_unit], distance_unit2mcp)
                #print([d for d in mcp_distribution if d >= distance_unit2mcp])
                #print(len(mcp_distribution))
                #print(mcp_distribution)

        return z_score_samples


def compute_p_value(z_scores):

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


def get_p_val(z_scores, period=30):

    import math
    p_val = np.arange(len(z_scores), dtype=np.float)

    for i in range(period):
        x = np.array(z_scores[:i])[~np.isnan(z_scores[:i])]
        p_val[i] = compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan

    for i in range(period, len(z_scores)):
        x = np.array(z_scores[i-period:i])[~np.isnan(z_scores[i-period:i])]
        #print(np.mean(x))
        print(i, x)
        print(i, x.size)
        print(i, np.mean(x))
        p_val[i] = compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan
        #averagedPval[i] = float(-math.log10(stats.t.sf(np.abs(tmp_d5), 3600-1)*2)) if not np.isnan(averagedPval[i]) else np.nan

        print(p_val[i])

    return p_val
