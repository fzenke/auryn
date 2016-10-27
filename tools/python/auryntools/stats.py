#!/usr/bin/python
import numpy as np
import pylab as pl
import struct

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

    ar = np.array(spikes)
    # Check for properties of correction formated list of spikes
    if len(ar.shape)!=2 or ar.shape[1]!=2:
        print("Invalid input format. Expected a list of spikes.")
        raise AttributeError

    t_min = ar[:,0].min()
    t_max = ar[:,0].max()
    t_diff = t_max-t_min

    nspikes = spike_counts(spikes)
    return 1.0*nspikes/t_diff

def rate_hist( spikes, *args, **kwargs ):
    return pl.hist(rates(spikes), *args, **kwargs )

def isis(spikes):
    '''
    Computes the ISI for spikes 
    '''

    last_spikes = np.zeros(1)
    ISIs = []
    for spike in spikes:
        t,i = spike
        if i>=len(last_spikes):
            last_spikes.resize(i+1)
        if last_spikes[i] > 0:
            ISIs.append(t-last_spikes[i])
        last_spikes[i] = t
    return ISIs

def isi_hist( spikes, *args, **kwargs ):
    return pl.hist(isis(spikes), *args, **kwargs )

def cvisis(spikes):
    '''
    Computes the CV ISI for spikes 
    '''

    last_spikes = np.zeros(1)
    sum1 = np.zeros(1)
    sum2 = np.zeros(1)
    nspikes = np.zeros(1)
    for spike in spikes:
        t,i = spike
        if i>=len(last_spikes):
            last_spikes.resize(i+1)
            sum1.resize(i+1)
            sum2.resize(i+1)
            nspikes.resize(i+1)
        if last_spikes[i] > 0:
            isi = t-last_spikes[i]
            sum1[i] += isi
            sum2[i] += isi**2
            nspikes[i] += 1
        last_spikes[i] = t

    cvisi_dist = []
    for i in xrange(len(sum1)):
        if nspikes[i]<2: continue
        mean = sum1[i]/nspikes[i]
        var  = sum2[i]/(nspikes[i]-1)-mean**2
        cvisi_dist.append(np.sqrt(var)/mean)

    return cvisi_dist

def cvisi_hist( spikes, *args, **kwargs ):
    return pl.hist(cvisis(spikes), *args, **kwargs )

def vogels_plot( spikes ):
    '''
    Quickly plots histograms for rate, ISI and CVISI like in Vogels et al. 2005
    '''

    pl.figure(figsize=(10,3))
    pl.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    pl.subplot(131)
    pl.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
    rate_hist(spikes, color="#00548c", bins=30)
    pl.xlabel("Rate [Hz]")

    pl.subplot(132)
    pl.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
    isi_hist(spikes, color="#ffcc00", bins=np.logspace(np.log10(1e-3), np.log10(10.0), 30))
    pl.gca().set_xscale("log")
    pl.xlabel("ISI [s]")

    pl.subplot(133)
    pl.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
    cvisi_hist(spikes, color="#c4000a", bins=30)
    pl.xlabel("CVISI [1]")
    pl.show()

