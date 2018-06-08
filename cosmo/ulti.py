__author__ = 'yuafan'

import numpy as np, matplotlib.pyplot as plt
import cosmo, math

def get_p_val(z_scores, period=30):

    p_val = np.arange(len(z_scores), dtype=np.float)

    for i in range(period):
        x = np.array(z_scores[:i])[~np.isnan(z_scores[:i])]
        p_val[i] = cosmo.compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan

    for i in range(period, len(z_scores)):
        x = np.array(z_scores[i-period:i])[~np.isnan(z_scores[i-period:i])]
        ### print(np.mean(x))
        ## print(i, x)
        ## print(i, x.size)
        ## print(i, np.mean(x))
        p_val[i] = cosmo.compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan
        #averagedPval[i] = float(-math.log10(stats.t.sf(np.abs(tmp_d5), 3600-1)*2)) if not np.isnan(averagedPval[i]) else np.nan

        ## print(p_val[i])

    return p_val


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
