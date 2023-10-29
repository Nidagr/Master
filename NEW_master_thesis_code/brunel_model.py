"""
This file contains the model of Brunel with inspiration from his article
https://link.springer.com/content/pdf/10.1023/A:1008925309027.pdf.
To make results comparable we use resolution 1/8 for all models.
"""
import nest
import pandas as pd
import numpy as np

"""
Version 1: The delays of connections between neurons are drawn from uniform distribution [1.0, 2.0]. 

Delay may have values [1.0,1.125,1.25,1.375,1.5,1.625,1.75,1.875,2.0]. 

If a=1, b=2, the end values 1.0 and 2.0 only get
half the probability compared to the rest.  The middle value can be rounded to from both sides, but the ends only
from one side. 

If a=0.9375, b=2.0625, we get equal probability for each value, even the end points.

We use iaf_psc_delta synapse, so exact spike times are rounded.
"""


def discrete_brunel(a=1.0, b=2.0, rounding=True, seed=123):
    """
    To get a model where delay end-points only have half the probability compared to inner points, set a=1.0, b=2.0.

    To make sure the endpoints have the same probability as the rest, set a=0.9375, b=2.0625.

    To round the spike times let rounding=True, to get exact spike times, let rounding=False.

    :param a: min value of uniform distribution from which delays are drawn from.
    :param b: max value of uniform distribution from which delays are drawn from.
    :param rounding: whether to round the spike times or not.
    :param seed: give other seed for nest if you want, default is 123.
    :return: pandas dataframes exc and inh of neuron ids and spike times. rate_ex, rate_in, firing rates of excitatory
    and inhibitory neuron populations, respectively.
    """
    nest.ResetKernel()
    nest.total_num_virtual_procs = 128  # increase computational power
    dt = 1 / 8  # resolution in ms
    simtime = 500.0  # duration of simulation
    g = 5.0  # see text
    eta = 2.0  # external rate relative to threshold rate
    epsilon = 0.1  # probability of connection
    num_neurons = 12500
    NE = int(0.8 * num_neurons)  # number of excitatory neurons
    NI = int(0.2 * num_neurons)  # number of inhibitory neurons

    N_rec = 500  # number of neurons to record from

    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron

    tauMem = 20.0  # time constant of membrane potential in ms
    theta = 20.0  # membrane threshold potential in mV

    neuron_params = {"C_m": 1.0,  # membrane capacitance
                     "tau_m": tauMem,  # membrane time constant
                     "t_ref": 2.0,  # refractory period time
                     "E_L": 0.0,  # leak reversal potential, here 0, because we have IF neurons, not LIF
                     "V_reset": 0.0,  # after spike, potential of neuron set back to this
                     "V_m": 0.0,  # initial membrane potential
                     "V_th": theta}  # threshold for spikes being fired

    J = 0.1  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

    # Neurons receive synapses from excitatory neurons outside the network, Poisson process with rate nu_ex
    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    nest.resolution = dt
    nest.rng_seed = seed

    if rounding:
        # Create the neurons
        nodes_ex = nest.Create("iaf_psc_delta", NE, params=neuron_params)
        nodes_in = nest.Create("iaf_psc_delta", NI, params=neuron_params)
        # the neurons receive poisson distributed input from neuron outside network
        noise = nest.Create("poisson_generator", params={"rate": p_rate})
    else:
        # Create the neurons
        nodes_ex = nest.Create("iaf_psc_delta_ps", NE, params=neuron_params)
        nodes_in = nest.Create("iaf_psc_delta_ps", NI, params=neuron_params)
        # the neurons receive poisson distributed input from neuron outside network
        noise = nest.Create("poisson_generator_ps", params={"rate": p_rate})

    # want to record spikes
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    # Connect "noise" to neurons
    nest.Connect(noise, nodes_ex, syn_spec={'weight': J_ex})  # gi konstant delay, blir satt konstant hvis ikke argument
    nest.Connect(noise, nodes_in, syn_spec={'weight': J_ex})

    # no delay for spike recorder
    # Record spikes from first N_rec neurons of each population, connect spike recorder to these neurons
    nest.Connect(nodes_ex[:N_rec], espikes)
    nest.Connect(nodes_in[:N_rec], ispikes)

    # create excitatory to inhibitory and excitatory-excitatory connections
    conn_params_ex = {'rule': 'fixed_indegree', 'indegree': CE}
    nest.Connect(nodes_ex, nodes_ex + nodes_in, conn_params_ex,
                 syn_spec={'weight': J_ex, 'delay': nest.random.uniform(a, b)})

    # create inhibitory to excitatory and inhibitory-inhibitory connections
    conn_params_in = {'rule': 'fixed_indegree', 'indegree': CI}
    nest.Connect(nodes_in, nodes_ex + nodes_in, conn_params_in,
                 syn_spec={'weight': J_in, 'delay': nest.random.uniform(a, b)})

    nest.Simulate(simtime)

    # total number of spikes from each population
    events_ex = espikes.n_events
    events_in = ispikes.n_events

    # get firing rates, number of spikes per second. Since we have in ms, need to divide spikes by simitime*1000
    # 1 s = 1000 ms, spikes / s = spikes / 1000 ms. Since we record from N_rec neurons, also divide by N_rec, as we
    # often talk about firing rates for a single neuron.
    rate_ex = events_ex / simtime * 1000 / N_rec
    rate_in = events_in / simtime * 1000 / N_rec

    exc_spikes = espikes.get('events')
    inh_spikes = ispikes.get('events')

    # dataframe containing spike times for first 500 neurons in each population
    exc = pd.DataFrame(exc_spikes)
    inh = pd.DataFrame(inh_spikes)

    return exc, inh, rate_ex, rate_in


"""
If you want to run the code :

# rounded spikes times
# for equal probability of each delay value:
exc, inh = discrete_brunel(a=0.9375, b=2.0625)

# for endpoints with half probabilities:
exc, inh = discrete_brunel()

# not rounded spike times
# for equal probability of each delay value:
exc, inh = discrete_brunel(a=0.9375, b=2.0625, rounding=False)

# for endpoints with half probabilities:
exc, inh = discrete_brunel(rounding=False)
"""

"""
Version 2: The Brunel model, but this time the delays are drawn from a continuous uniform distribution [1.0, 2.0].
The probability of drawing each number is the same. To be able to do this we use a different type of synapse, making the
spike times exact and not rounded. Needed to be taken into account when comparing with discrete models above.
"""


def continuous_brunel(seed=123):
    """
    :param seed: give different value for seed, default is 123.
    :return: pandas dataframes exc and inh of spike times of neurons on excitatory and inhibitory neurons respectively.
    rate_ex and rate_in are the firing rates of each of the populations.
    """
    nest.ResetKernel()

    dt = 1 / 8  # resolution in ms
    simtime = 500.0  # duration of simulation
    nest.total_num_virtual_procs = 128  # increase computational power
    g = 5.0  # see text
    eta = 2.0  # external rate relative to threshold rate
    epsilon = 0.1  # probability of connection
    num_neurons = 12500
    NE = int(0.8 * num_neurons)  # number of excitatory neurons
    NI = int(0.2 * num_neurons)  # number of inhibitory neurons

    N_rec = 500  # number of neurons to record from

    CE = int(epsilon * NE)  # number of excitatory synapses per neuron
    CI = int(epsilon * NI)  # number of inhibitory synapses per neuron
    C_tot = int(CI + CE)  # total number of synapses per neuron

    tauMem = 20.0  # time constant of membrane potential in ms
    theta = 20.0  # membrane threshold potential in mV

    neuron_params = {"C_m": 1.0,  # membrane capacitance
                     "tau_m": tauMem,  # membrane time constant
                     "t_ref": 2.0,  # refractory period time
                     "E_L": 0.0,  # leak reversal potential, here 0, because we have IF neurons, not LIF
                     "V_reset": 0.0,  # after spike, potential of neuron set back to this
                     "V_m": 0.0,  # initial membrane potential
                     "V_th": theta}  # threshold for spikes being fired

    J = 0.1  # postsynaptic amplitude in mV
    J_ex = J  # amplitude of excitatory postsynaptic potential
    J_in = -g * J_ex  # amplitude of inhibitory postsynaptic potential

    # Neurons receive synapses from excitatory neurons outside the network, Poisson process with rate nu_ex
    nu_th = theta / (J * CE * tauMem)
    nu_ex = eta * nu_th
    p_rate = 1000.0 * nu_ex * CE

    nest.resolution = dt
    nest.rng_seed = seed

    # Create the neurons
    nodes_ex = nest.Create("iaf_psc_delta_ps", NE, params=neuron_params)
    nodes_in = nest.Create("iaf_psc_delta_ps", NI, params=neuron_params)

    # want to record spikes
    espikes = nest.Create("spike_recorder")
    ispikes = nest.Create("spike_recorder")

    # the neurons receive poisson distributed input from neuron outside network
    noise = nest.Create("poisson_generator_ps", params={"rate": p_rate})

    # Connect "noise" to neurons
    nest.Connect(noise, nodes_ex, syn_spec={'weight': J_ex})
    nest.Connect(noise, nodes_in, syn_spec={'weight': J_ex})

    # Record spikes from first N_rec neurons of each population, connect spike recorder to these neurons
    nest.Connect(nodes_ex[:N_rec], espikes)
    nest.Connect(nodes_in[:N_rec], ispikes)

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
    conn_ex.set(delay=np.random.uniform(1.0, 2.0, nc_ex))

    nc_in = len(conn_in)
    conn_in.set(delay=np.random.uniform(1.0, 2.0, nc_in))

    nest.Simulate(simtime)

    # total number of spikes from each population
    events_ex = espikes.n_events
    events_in = ispikes.n_events

    # get firing rates, number of spikes per second. Since we have in ms, need to divide spikes by simitime*1000
    # 1 s = 1000 ms, spikes / s = spikes / 1000 ms. Since we record from N_rec neurons, also divide by N_rec, as we
    # often talk about firing rates for a single neuron.
    rate_ex = events_ex / simtime * 1000 / N_rec
    rate_in = events_in / simtime * 1000 / N_rec

    exc_spikes = espikes.get('events')
    inh_spikes = ispikes.get('events')

    # dataframe containing spike times for first 500 neurons in each population
    exc = pd.DataFrame(exc_spikes)
    inh = pd.DataFrame(inh_spikes)

    return exc, inh, rate_ex, rate_in


"""
To get the Brunel model with continuous delays:
exc, inh = continuous_brunel()
"""
