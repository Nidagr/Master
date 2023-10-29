"""

We want to retrieve the CV distributions of the spike data from the 5 min simulation of the Brunel model
with rounded spike times and delays drawn from discrete delay interval [1.0, 2.0].
"""

import nest
import numpy as np
import pandas as pd
import elephant
import quantities as pq
import neo
import sys
from scipy.stats import ks_2samp
import scipy.stats as stats


def get_cvs(spike_data):
    """
    Get the CV for each neuron recorded.

    CV = standard_deviation(ISIs)/mean(ISIs)
    """
    cvs = []

    spike_data = spike_data.sort_values(by='time_ms')
    grouped = spike_data.groupby(spike_data['sender'])

    for name, group in grouped:
        """
        Each group is senders and times for one value of senders. That is, we iterate through all 
        neurons. And the times for each neuron is in sorted order. Therefore, the cvs
        returned must have the same order. So cvs contain cv of neuron 1, then neuron 2 .... then neuron N.
        """
        t = np.asarray(group['time_ms'])
        spiketrain = neo.core.SpikeTrain(t * pq.ms, t_start=0 * pq.ms, t_stop=300000 * pq.ms)
        isi = elephant.statistics.isi(spiketrain)
        cv = elephant.statistics.cv(isi)
        cvs.append(cv)

    return cvs


def get_cv_lists(exc, inh):
    """
    Get the CV distribution of each simulation and return lists of lists of CV distributions.
    DO so for both excitatory and inhibitory neurons.
    """
    cv_list_exc = list()
    for i in range(1, 11):
        exc_cvs = get_cvs(exc[i])
        cv_list_exc.append(exc_cvs)

    cv_list_inh = list()
    for i in range(1, 11):
        inh_cvs = get_cvs(inh[i])
        cv_list_inh.append(inh_cvs)
    return cv_list_exc, cv_list_inh


def round_1_2():
    """
    Get spike data from brunel model with rounded spike times and delays drawn from discrete interval [1.0, 2.0].
    Simulated for 5 min.
    """
    #spike_path = '/users/ngrnbekk/spike_data/5min'
    spike_path = '/p/home/jusers/groenbekk1/jusuf/project/spike_data/5min'
    exc = {}
    for i in range(1, 11):
        exc[i] = pd.read_csv(r'{}/brunel_rounding_True_delay_1.0_2.0_seed_{}_spikes_exc-12502-0.dat'.format(spike_path, i),
                             skiprows=2, sep='\t')

    inh = {}
    for i in range(1, 11):
        inh[i] = pd.read_csv(r'{}/brunel_rounding_True_delay_1.0_2.0_seed_{}_spikes_inh-12503-0.dat'.format(spike_path, i),
                             skiprows=2, sep='\t')
    return exc, inh


exc, inh = round_1_2()
cv_list_exc, cv_list_inh = get_cv_lists(exc, inh)
df_exc = pd.DataFrame(cv_list_exc)
df_inh = pd.DataFrame(cv_list_inh)
df_exc.to_csv('brunel_round_1_2_5min_CV_exc.csv')
df_inh.to_csv('brunel_round_1_2_5min_CV_inh.csv')
