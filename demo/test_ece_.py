__author__ = 'yuafan'

#test_data.xlsx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import cosmo.ulti as ulti
import sys
if not( u'../' in sys.path): sys.path.append( u'../' )

import cosmo.cosmo as cosmo
reload(cosmo)


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

def get_zscore_buswise(z_score):

    z_scores_buswise = [[] for ii in range(n_unit)]

    for idays in range(len(z_score)):

        for ibus in range(len(z_score[idays])):

            z_scores_buswise[ibus].append(z_score[idays][ibus])

    return z_scores_buswise


data_dir = '../sample_data/ece_test_data.xlsx'

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

model_samples_ = get_models(df_flow, sample_size=24, sample_id=sample_id)
model_samples_tw = get_samples_time_interval_wise(model_samples_)

n_unit, n_var = len(sample_id), 7

z_scores, p_val_timeline = [], []

for idays in range(n_var, len(tl)):

    ## print(idays)
    p_val_timeline.append(tl[idays])
    group_tmp = cosmo.Anomaly(n_unit, n_var)
    set_samples(model_samples_tw[idays-n_var:idays], group_tmp)
    group_tmp.set_distance(distance='Hellinger')
    group_tmp.compute_distance_matrix()
    z_scores.append(group_tmp.get_z_score()[:])

z_scores_buswise = get_zscore_buswise(z_scores)

p_val = []

for ibus in range(n_unit):

    p_val.append(ulti.get_p_val(z_scores_buswise[ibus]))

ulti.plot_allrepair(p_val, p_val, p_val_timeline)
ulti.plot_all_z_score(z_scores_buswise, z_scores_buswise, p_val_timeline)


