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
import matplotlib.pyplot as plt

################################################################################
def plot_allrepair(P_value, P_value_delta, timeline, info='', thresh_ = 5):

    idx_ = range(12)
    plt.clf()

    f, axarr = plt.subplots(12, 1, figsize=(11, 20), dpi=500, facecolor='w', edgecolor='k')

    for i in range(len(idx_)):
        cr = ['r', 'b', 'k', 'k', 'g', 'g', 'y', 'y']

        Pval = P_value[i]
        Pval_del = P_value_delta[i]

        xdat = timeline[:]

        if len(timeline) != len(Pval):
            xdat = timeline[6:]

        ## print(len(xdat), len(Pval))


        ss = timeline[0]
        ee = timeline[-1]

        r = i
        axarr[r].scatter(xdat, Pval, color='c', s = 3,  marker='D')
        axarr[r].scatter(xdat, Pval_del, color='b', s = 3,  marker='D')

        #axarr[r].set_ylabel('-log10(P-Value for Mean)')
        axarr[r].set_ylabel(idx_[i])
        #axarr[r].set_xlim([dt.getData(ss),dt.getData(ee)])
        axarr[r].set_xlim([ss,ee])
        axarr[r].set_ylim([0, 20])
        #axarr[r].axhline(y = thresh_, color='#cdc9c9')

    f.tight_layout()
    fname = 'ece_test_2am_z_energy.png'
    plt.savefig(fname)

################################################################################
def plot_all_z_score(P_value, P_value_delta, timeline, info='', thresh_ = 5):

    idx_ = range(12)
    plt.clf()
    f, axarr = plt.subplots(12, 1, figsize=(15, 20), dpi=500, facecolor='w', edgecolor='k')

    for i in range(len(idx_)):

        cr = ['r', 'b', 'k', 'k', 'g', 'g', 'y', 'y']

        Pval = P_value[i]
        Pval_del = P_value_delta[i]

        xdat = timeline[:]

        if len(timeline) != len(Pval):
            xdat = timeline[6:]

        ## print(len(xdat), len(Pval))

        #ss = tu.get_delta_ms(tu.gen_datetime_ymd(2011,6,16))
        #ee = tu.get_delta_ms(tu.gen_datetime_ymd(2015,10,8))

        ss = timeline[0]
        ee = timeline[-1]

        r = i
        axarr[r].scatter(xdat, Pval, color='c', s = 3,  marker='D')
        axarr[r].scatter(xdat, Pval_del, color='b', s = 3,  marker='D')

        #axarr[r].set_ylabel('-log10(P-Value for Mean)')
        axarr[r].set_ylabel(idx_[i])
        #axarr[r].set_xlim([dt.getData(ss),dt.getData(ee)])
        axarr[r].set_xlim([ss,ee])
        axarr[r].set_ylim([0, 1])
        #axarr[r].axhline(y = thresh_, color='#cdc9c9')

    f.tight_layout()
    fname = 'ece_test_2am_z_energy.png'
    plt.savefig(fname)
