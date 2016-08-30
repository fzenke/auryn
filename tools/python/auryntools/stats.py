#!/usr/bin/python
import numpy as np
import pylab as pl
import struct

def cvisi_hist( spikes, *args, **kwargs ):
    pl.hist(cvisi(spikes), *args, **kwargs )

def spike_counts(spikes):
    '''
    Compute spike count for each neuron
    '''

    nspikes = np.zeros(1)
    for spike in spikes:
        t,i = spike
        if i>=len(nspikes):
            nspikes.resize(i+1)
        nspikes[i] += 1
    return nspikes

def rates(spikes):
    '''
    Compute firing rates for each neuron
    '''

    nspikes = spike_counts(spikes)
    ar = np.array(spikes)
    t_min = ar[:,0].min()
    t_max = ar[:,0].max()
    t_diff = t_max-t_min
    return 1.0*nspikes/t_diff

def rate_hist( spikes, *args, **kwargs ):
    pl.hist(rates(spikes), *args, **kwargs )

