#!/usr/bin/python
import numpy as np
import pylab as pl
from auryntools import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with mpirun and default paramters. 
# This generates spk output files under /tmp/

num_mpi_ranks = 4
seconds = 0.1

filenames  = [ "/tmp/coba.%i.e.spk"%i for i in range(num_mpi_ranks) ]

sf = AurynBinarySpikeView(filenames)
spikes = np.array(sf.get_last(seconds))

pl.scatter(spikes[:,0], spikes[:,1])
pl.xlabel("Time [s]")
pl.ylabel("Neuron ID")
pl.show()
    


