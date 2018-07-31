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

from __future__ import print_function, division

import numpy as np
import metrics as metrics

################################################################################
class Anomaly(object):
    '''TODO:

    Samples with its observations that came from a population.

    '''

    ########################################
    def __init__(self, n_units, n_variations):
        ''' TODO:
        <short description of the method>
        
        Inputs
        ------
        n_units: integer, number of samples
        n_variations: integer, number of observations from each sample

        Usage Example
        -------------
        
        '''
        self.n_units = n_units
        self.n_variations = n_variations

        #self.models_samples = []
        #self.models_samples = [[[] for var in range(n_variations)] for unit in range(n_units)]
        self.models_samples = [[[] for unit in range(n_units)] for var in range(n_variations)]

        self.distance = None
        self.distance_matrix = None

    ########################################
    def set_samples(self, samples):
        ''' TODO:
        <short description of the method>
        
        Input
        -----
        samples: list of lists, observations of samples

        Usage Example
        -------------
        '''

        self.models_samples = samples

    ########################################
    def set_distance(self, distance='Hellinger'):
        ''' TODO:
        <short description of the method>

        Set type of metric for comparing two feature vectors

        Paramters
        ---------

        distance: string, e.g. 'Hellinger', 'EMD', 'Euclidean', 'Cityblock' ...
        
        Usage Example
        -------------
        
        '''
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

    ########################################
    def compute_distance_matrix(self):
        ''' TODO:
        Compute pair-wise distance matrix for all pairs of sample observations

        
        Usage Example
        -------------
        
        '''

        symmetrize = lambda a: a + a.T - np.diag(a.diagonal())

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

        
    ########################################
    def get_z_score(self):
        ''' TODO:
        Compute z score based on distance matrix, Most Central Pattern mode

        TODO: add knn mode
                
        Usage Example
        -------------
        
        '''
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

