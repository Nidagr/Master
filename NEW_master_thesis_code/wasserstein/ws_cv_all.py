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

class CV_test_class(sciunit.TestM2M, tests.isi_variation_test):
    default_params = {'variation_measure': 'cv'} # need to set cv as lvr is default in tests.isi_variation_test
    score_type = scores.wasserstein_distance
    
CV_test = CV_test_class()


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

    for inst in list_of_instances:
        CV_scores = CV_test.judge([cont,inst])
        v = CV_scores.score.iloc[1,0]
        ws_distances.append(v)
    return ws_distances

if len(sys.argv) != 4:
    raise RuntimeError('3 arguments required')
    
kind = sys.argv[1] # droop or equal
res = int(sys.argv[2]) # 2,4,8,...
seed = int(sys.argv[3])

base = '/p/home/jusers/groenbekk1/jusuf/project/wasserstein_calculations/organized_spike_data'
# Create the continuous one, seed 1
cont_seed_1 = f'{base}/resolution_1_8/brunel_continuous/brunel_continuous_delay_1.0_2.0_seed_1_spikes_exc-12502-0.dat'
cont_1 = create_class(cont_seed_1)
cont = cont_1()
cont.produce_spiketrains();

mind = 1.0-1/res/2
maxd = 2.0+1/res/2

if kind == 'droop':
# if equal, or droop...
# file paths all models
    droop_1_4 = [f'{base}/resolution_1_{res}/brunel_rounding_1_2/brunel_rounding_True_delay_1.0_2.0_seed_{seed}_spikes_exc-12502-0.dat']
else:
    droop_1_4 = [f'{base}/resolution_1_{res}/brunel_rounding_1_2/brunel_rounding_True_delay_{mind}_{maxd}_seed_{seed}_spikes_exc-12502-0.dat']
# Create classes for each type of model
droop_1_4_classes = get_classes(droop_1_4)

# Create instances of each class
droop_1_4_instances = create_instances(droop_1_4_classes)

# Get the Wasserstein distances
ws_droop_1_4 = get_ws_distances(droop_1_4_instances)



print(ws_droop_1_4)
t = pd.DataFrame(ws_droop_1_4, index = [seed])
t.to_csv(f'ws_cv_{kind}_1_{res}_seed_{seed}.csv')
