__author__ = 'yuafan'

#test_data.xlsx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data_dir = '/Users/yuafan/Desktop/SideProj/ece/test_data.xlsx'

df_flow = pd.read_excel(data_dir, 'Energy', skiprows=1, index_col='Row Labels')
df_flow.fillna(0,inplace=True)

df_flow.index = pd.to_datetime(df_flow.index)
df_flow.index.names = ['Date']

sample_id = ['735999124006020247',
             '735999124006029158',
             '735999124006031489',
             '735999124006033902',
             '735999124006037559',
             '735999124006117015',
             '735999124006117022',
             '735999124006117039',
             '735999124006124266',
             '735999124006125478',
             '735999156006005986',
             '735999156006019525']

timeline = df_flow['735999124006020247'].index

# computing models

tl = timeline.tolist()[::24]
del tl[-1] #...

def get_models(df_flow_dict, sample_size=24, sample_id=[]):

    model_samples = [[] for isamples in range(len(sample_id))]

    for i_interval in range(int(len(timeline)/sample_size)):

        for i_sample, sample in enumerate(sample_id):

            sample_tmp = df_flow[sample][i_interval*sample_size:i_interval*sample_size+sample_size]
            model_samples[i_sample].append([np.mean(sample_tmp), np.std(sample_tmp)])

    return model_samples


def get_samples_time_interval_wise(model_pack):

    model_pack_tw = [[] for i_int in range(len(model_pack[0]))]

    for i_sample in range(len(model_pack)):

        for i_int in range(len(model_pack[i_sample])):

            model_pack_tw[i_int].append(model_pack[i_sample][i_int])

    return model_pack_tw

model_samples_ = get_models(df_flow, sample_size=24, sample_id=sample_id)
model_samples_tw = get_samples_time_interval_wise(model_samples_)

import cosmo.cosmo as cosmo

n_unit, n_var = len(sample_id), 7

def set_samples(histogram_pack, anomaly):

    for isamples in range(anomaly.n_units):

        for idays in range(anomaly.n_variations):

            anomaly.models_samples[idays][isamples] = histogram_pack[idays][isamples][:]

    return histogram_pack

def set_samples_district_heating(model_pack, anomaly):

    for isamples in range(anomaly.n_units):

        for idays in range(anomaly.n_variations):

            anomaly.models_samples[idays][isamples] = model_pack[isamples][idays][:]

    return model_pack



z_scores, p_val_timeline = [], []

for idays in range(n_var, len(tl)):

    print(idays)
    p_val_timeline.append(tl[idays])
    group_tmp = cosmo.Anomaly(n_unit, n_var)
    set_samples(model_samples_tw[idays-n_var:idays], group_tmp)
    group_tmp.set_distance(distance='Hellinger')
    group_tmp.compute_distance_matrix()
    z_scores.append(group_tmp.get_z_score()[:])


def get_zscore_buswise(z_score):

    z_scores_buswise = [[] for ii in range(n_unit)]

    for idays in range(len(z_score)):

        for ibus in range(len(z_score[idays])):

            z_scores_buswise[ibus].append(z_score[idays][ibus])

    return z_scores_buswise

z_scores_buswise = get_zscore_buswise(z_scores)

p_val = []

def get_p_val(z_scores, period=30):

    import math
    p_val = np.arange(len(z_scores), dtype=np.float)

    for i in range(period):
        x = np.array(z_scores[:i])[~np.isnan(z_scores[:i])]
        p_val[i] = cosmo.compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan

    for i in range(period, len(z_scores)):
        x = np.array(z_scores[i-period:i])[~np.isnan(z_scores[i-period:i])]
        #print(np.mean(x))
        print(i, x)
        print(i, x.size)
        print(i, np.mean(x))
        p_val[i] = cosmo.compute_p_value(x) if x.size >= 1 else np.nan
        p_val[i] = float(-math.log10(p_val[i])) if not np.isnan(p_val[i]) else np.nan
        #averagedPval[i] = float(-math.log10(stats.t.sf(np.abs(tmp_d5), 3600-1)*2)) if not np.isnan(averagedPval[i]) else np.nan

        print(p_val[i])

    return p_val

for ibus in range(n_unit):

    p_val.append(get_p_val(z_scores_buswise[ibus]))



def plot_allrepair(P_value, P_value_delta, timeline, info='', thresh_ = 5):

    import matplotlib.pyplot as plt
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

        print(len(xdat), len(Pval))


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
    plt.savefig('/Users/yuafan/Desktop/busData/ece_test_2am_energy.png')


def plot_all_z_score(P_value, P_value_delta, timeline, info='', thresh_ = 5):

    import matplotlib.pyplot as plt
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

        print(len(xdat), len(Pval))

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
    plt.savefig('/Users/yuafan/Desktop/busData/ece_test_2am_z_energy.png')


plot_allrepair(p_val, p_val, p_val_timeline)
plot_all_z_score(z_scores_buswise, z_scores_buswise, p_val_timeline)




# check distribution

# total: 105131
# 8672 of 0s

'''
all_values = []
for isamples in sample_id:
    all_values.extend(df_flow[isamples].values.tolist())

stat_val = sorted(all_values)[:]



def getHist(value, bin_=100, range_=[0,100], info=' '):
    # Normalized_ return numpy arrays
    #count, bins, ignored = plt.hist(_val, 60, normed=True, facecolor='blue')  # color... = = really?
    count, bins = np.histogram(value, bins=bin_, range=range_, density=True, normed=True)
    count = count * (bins[1] - bins[0]) # a must for normalization
    center = (bins[:-1] + bins[1:]) / 2
    width = 0.7 * (bins[1] - bins[0])
    plt.clf()
    plt.bar(center, count, align='center', width=width, alpha = 0.6)
    #plt.axis([0, 12, 0, 1])
    plt.xlabel('Value')
    plt.ylabel('Volume')
    #plt.title(_elem + ' Sample: ' + str(_span/3600) + 'hours, Starting date:' + str(tiop.getData(_timebase)))
    #plt.savefig(_path + _elem +'.png')
    plt.tight_layout()
    plt.show()
    plt.clf()
    return count, center, width

count= getHist(stat_val, bin_=200, range_=[0,55])


'''