#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from auryntools import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with mpirun and default paramters. 
# This generates spk output files under /tmp/

num_mpi_ranks = 4
seconds = 0.1

filenames  = [ "/tmp/coba.%i.e.spk"%i for i in range(num_mpi_ranks) ]

sf = AurynBinarySpikeView(filenames)
spikes = np.array(sf.get_last(seconds))

plt.scatter(spikes[:,0], spikes[:,1])
plt.xlabel("Time [s]")
plt.ylabel("Neuron ID")
plt.show()
    


