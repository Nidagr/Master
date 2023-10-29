import neo
from networkunit import tests, scores, models
from networkunit.capabilities.ProducesSpikeTrains import ProducesSpikeTrains # branch v0.2
from copy import copy
import pandas as pd
import numpy as np
import elephant
import quantities as pq
import sciunit
from quantities import ms
import sys
sys.setrecursionlimit(10000)
from pathlib import Path

def create_class(path):
    """
    This function creates a model class for the specific file path given.
    """
    class model(models.loaded_spiketrains): 
    
        default_params = {'file_path':path}

        def load(self):
            file_path = self.default_params['file_path']        
            data = pd.read_csv(file_path,skiprows=2,sep='\t')
            spike_data = data.sort_values(by='time_ms')
            grouped = spike_data.groupby(spike_data['sender'])

            spike_trains = []
            for name, group in grouped:
                    """
                    Each group is senders and times for one value of senders. That is, we iterate through all 
                    neurons. And the times for each neuron is in sorted order. Therefore, the cvs
                    returned must have the same order. So cvs contain cv of neuron 1, then neuron 2 .... then neuron N.
                    """
                    t = np.asarray(group['time_ms'])
                    spiketrain = neo.core.SpikeTrain(t * pq.ms, t_start=0*pq.ms, t_stop=10000*pq.ms)
                    spike_trains.append(spiketrain)
            return spike_trains
    return model

def get_classes(path_list):
    classes = []
    for p in path_list:
        cl = create_class(p)
        classes.append(cl)
    return classes

class FR_test_class(sciunit.TestM2M, tests.firing_rate_test):
    score_type = scores.wasserstein_distance
    
FR_test = FR_test_class()


def create_instances(class_list):
    """
    Iterate through list of classes, create instance and run create_spiketrains(). Return list of the instances, 
    ready to be measured.
    """
    instances = []
    for cl in class_list:
        c = cl()
        c.produce_spiketrains();
        instances.append(c)
    return instances

def get_ws_distances(list_of_instances):
    """
    Run the wasserstein metric on continuous model vs all seeds one model one resolution by passing list of 
    instances of class.
    """
    ws_distances = [] 

    for i in range(0,10):
        FR_scores = FR_test.judge([cont,list_of_instances[i]])
        v = FR_scores.score.iloc[1,0]
        ws_distances.append(v)
    return ws_distances

# Create the continuous one, seed 1
cont_seed_1 = '/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_8/brunel_continuous/brunel_continuous_delay_1.0_2.0_seed_1_spikes_exc-12502-0.dat'
cont_1 = create_class(cont_seed_1)
cont = cont_1()
cont.produce_spiketrains();

# file paths all models
droop_1_2 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_2/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_4 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_4/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_8 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_8/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_16 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_16/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_32 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_32/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_64 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_64/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_128 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_128/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]
droop_1_256 = [f'/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data/resolution_1_256/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{i}_spikes_exc-12502-0.dat'.format(i) for i in range(1,11)]

# Create classes for each type of model
droop_1_2_classes = get_classes(droop_1_2)
"""droop_1_4_classes = get_classes(droop_1_4)
droop_1_8_classes = get_classes(droop_1_8)
droop_1_16_classes = get_classes(droop_1_16)
droop_1_32_classes = get_classes(droop_1_32)
droop_1_64_classes = get_classes(droop_1_64)
droop_1_128_classes = get_classes(droop_1_128)
droop_1_256_classes = get_classes(droop_1_256)
"""
# Create instances of each class
droop_1_2_instances = create_instances(droop_1_2_classes)
"""droop_1_4_instances = create_instances(droop_1_4_classes)
droop_1_8_instances = create_instances(droop_1_8_classes)
droop_1_16_instances = create_instances(droop_1_16_classes)
droop_1_32_instances = create_instances(droop_1_32_classes)
droop_1_64_instances = create_instances(droop_1_64_classes)
droop_1_128_instances = create_instances(droop_1_128_classes)
droop_1_256_instances = create_instances(droop_1_256_classes)
"""
# Get the Wasserstein distances
ws_droop_1_2 = get_ws_distances(droop_1_2_instances)
"""ws_droop_1_4 = get_ws_distances(droop_1_4_instances)
ws_droop_1_8 = get_ws_distances(droop_1_8_instances)
ws_droop_1_16 = get_ws_distances(droop_1_16_instances)
ws_droop_1_32 = get_ws_distances(droop_1_32_instances)
ws_droop_1_64 = get_ws_distances(droop_1_64_instances)
ws_droop_1_128 = get_ws_distances(droop_1_128_instances)
ws_droop_1_256 = get_ws_distances(droop_1_256_instances) """


print(ws_droop_1_2)
t = pd.DataFrame(ws_droop_1_2, index = [1,2,3,4,5,6,7,8,9,10])
t.to_csv('ws_droop_1_2')