#!/usr/bin/python
import numpy as np
import pylab as pl
from auryntools import *

# This example assumes that you have run the code from
# https://github.com/fzenke/pub2015orchestrated/
# using BinarySpikeMonitors (the development branch does that per default)

datadir = "/home/zenke/data/sim" # Set this to your data path
num_mpi_ranks = 4

dim = 64
n_max  = dim**2
t_bin  = 100e-3
integration_time = 400
neuron_id  = 28

outputfile =  "%s/rf2.0.e.spk"%datadir
sf = AurynBinarySpikeFile(outputfile)

stimfiles  = ["%s/rf2.%i.s.spk"%(datadir,i) for i in range(num_mpi_ranks)]
sfo = AurynBinarySpikeView(stimfiles)

start_times = np.arange(6)*500
for i,t_start in enumerate(start_times):
    t_end   = t_start+integration_time
    print("Analyzing %is..%is"%(t_start,t_end))
    spike_times = np.array(sf.get_spike_times(neuron_id, t_start, t_end))
    hist = sfo.time_triggered_histogram( spike_times, time_offset=-t_bin, time_window=t_bin, max_neuron_id=n_max )
    pl.subplot(2,3,i+1)
    pl.title("t=%is"%t_start)
    pl.imshow(hist.reshape((dim,dim)), origin='bottom')
pl.show()


    


