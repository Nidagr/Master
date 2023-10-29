"""
Brunel model
-------------------------------------
Try to make own version of Brunel using the template given by beNNch to make sure
simulations can be run.

Record all neurons. Continuous delays.
"""

import numpy as np
import os
import time
import scipy.special as sp

import nest
import nest.raster_plot

# Parameters, need these for beNNch to work on our code--------------------------------------------------

params = {
    'nvp': {num_vps},  # total number of virtual processes
    'scale': {N_SCALING},  # scaling factor of the network size
    # total network size = scale*11250 neurons
    'simtime': {model_time_sim},  # total simulation time in ms
    'presimtime': {model_time_presim},  # simulation time until reaching equilibrium
    'dt': 1 / 8,  # simulation step
    'rng_seed': {rng_seed},  # random number generator seed
    'path_name': '.',  # path where all files will have to be written
    'log_file': 'logfile',  # naming scheme for the log files
    'min_delay': {mind},  # minimum value for delay
    'max_delay': {maxd},  # maximum value for delay
}
# -------------------------------------------------------------------------------------------------------------


def build_network():

    tic = time.time()  # start timer on construction
    # set global kernel parameters
    nest.SetKernelStatus({'total_num_virtual_procs': params['nvp'],
                          'resolution': params['dt'],
                          'tics_per_ms': 2**10,
                          'rng_seed': params['rng_seed'],
                          'overwrite_files': True})
    tauMem = 20.0  # time constant of membrane potential in ms
    theta = 20.0  # membrane threshold potential in mV

    neuron_params = {"C_m": 1.0,  # membrane capacitance
                     "tau_m": tauMem,  # membrane time constant
                     "t_ref": 2.0,  # refractory period time
                     "E_L": 0.0,  # leak reversal potential, here 0, because we have IF neurons, not LIF
                     "V_reset": 0.0,  # after spike, potential of neuron set back to this
                     "V_m": 0.0,  # initial membrane potential
                     "V_th": theta}  # threshold for spikes being fired

    g = 5.0  # see text
    eta = 2.0  # external rate relative to threshold rate
    epsilon = 0.1  # probability of connection
    num_neurons = 12500*params['scale']
    NE = int(0.8 * num_neurons)  # number of excitatory neurons
    NI = int(0.2 * num_neurons)  # number of inhibitory neurons

    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron

    J = 0.1  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

    # Neurons receive synapses from excitatory neurons outside the network, Poisson process with rate nu_ex
    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    # Create the neurons
    nodes_ex = nest.Create("iaf_psc_delta_ps", NE, params=neuron_params)
    nodes_in = nest.Create("iaf_psc_delta_ps", NI, params=neuron_params)
    # the neurons receive poisson distributed input from neuron outside network
    noise = nest.Create("poisson_generator_ps", params={"rate": p_rate})

    # want to record spikes
    recorder_label = os.path.join(
        params['path_name'],
        'brunel_continuous' + '_delay_' + str(params['min_delay']) + '_' +
        str(params['max_delay']) + '_seed_' + str(params['rng_seed']) + '_spikes_exc')
    espikes = nest.Create("spike_recorder", params={
        'record_to': 'ascii',
        'label': recorder_label,
        'precision': 12
    })

    recorder_label_i = os.path.join(
        params['path_name'],
        'brunel_continuous_' + '_delay_' + str(params['min_delay']) + '_' +
        str(params['max_delay']) + '_seed_' + str(params['rng_seed']) + '_spikes_inh')
    ispikes = nest.Create("spike_recorder", params={
        'record_to': 'ascii',
        'label': recorder_label_i,
        'precision': 12
    })

    BuildNodeTime = time.time() - tic  # Time it took to create neurons and spike recorders
    node_memory = str(memory_thisjob())

    tic = time.time()

    # Connect "noise" to neurons
    nest.Connect(noise, nodes_ex, syn_spec={'weight': J_ex})
    nest.Connect(noise, nodes_in, syn_spec={'weight': J_ex})

    # no delay for spike recorder
    # Want to record ALL neurons
    nest.Connect(nodes_ex, espikes)
    nest.Connect(nodes_in, ispikes)

    # create excitatory to inhibitory and excitatory-excitatory connections
    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex,
                 syn_spec={'weight': J_ex, 'synapse_model': 'cont_delay_synapse'})

    # create inhibitory to excitatory and inhibitory-inhibitory connections
    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in,
                 syn_spec={'weight': J_in, 'synapse_model': 'cont_delay_synapse'})

    conn_ex = nest.GetConnections(source=nodes_ex, synapse_model='cont_delay_synapse')
    conn_in = nest.GetConnections(source=nodes_in, synapse_model='cont_delay_synapse')

    # Change the delays of the neuron-neuron connections to be continuous
    nc_ex = len(conn_ex)
    conn_ex.set(delay=np.random.uniform(params['min_delay'], params['max_delay'], nc_ex))

    nc_in = len(conn_in)
    conn_in.set(delay=np.random.uniform(params['min_delay'], params['max_delay'], nc_in))

    BuildEdgeTime = time.time() - tic  # time it took to connect all edges
    network_memory = str(memory_thisjob())

    d = {'py_time_create': BuildNodeTime,
         'py_time_connect': BuildEdgeTime,
         'node_memory': node_memory,
         'network_memory': network_memory}

    return d, espikes, ispikes


def run_simulation():
    """Performs a simulation, including network construction"""

    nest.ResetKernel()

    base_memory = str(memory_thisjob())

    build_dict, sr_ex, sr_in = build_network()

    tic = time.time()

    nest.Simulate(params['presimtime'])

    PreparationTime = time.time() - tic
    init_memory = str(memory_thisjob())

    tic = time.time()

    nest.Simulate(params['simtime'])

    SimCPUTime = time.time() - tic
    total_memory = str(memory_thisjob())

    average_rate_ex = compute_rate(sr_ex)
    average_rate_in = compute_rate(sr_in)

    d = {'py_time_presimulate': PreparationTime,
         'py_time_simulate': SimCPUTime,
         'base_memory': base_memory,
         'init_memory': init_memory,
         'total_memory': total_memory,
         'average_rate_ex': average_rate_ex,
         'average_rate_in': average_rate_in}
    d.update(build_dict)
    d.update(nest.GetKernelStatus())
    print(d)

    fn = '{fn}_{rank}.dat'.format(fn=params['log_file'], rank=nest.Rank())
    with open(fn, 'w') as f:
        for key, value in d.items():
            f.write(key + ' ' + str(value) + '\n')


def compute_rate(sr):
    """Compute local approximation of average firing rate

    This approximation is based on the number of local nodes, number
    of local spikes and total time. Since this also considers devices,
    the actual firing rate is usually underestimated.

    """

    n_local_spikes = sr.n_events
    n_local_neurons = 12500*params['scale']
    simtime = params['simtime']
    return 1. * n_local_spikes / (n_local_neurons * simtime) * 1e3


def memory_thisjob():
    """Wrapper to obtain current memory usage"""
    nest.ll_api.sr('memory_thisjob')
    return nest.ll_api.spp()


if __name__ == '__main__':
    run_simulation()
