#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from auryntools import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with default paramters. 
# This generates spk output files under /tmp/


filename  =  "/tmp/coba.0.e.spk"
seconds = 0.1

sf = AurynBinarySpikeFile(filename)
spikes = np.array(sf.get_last(seconds))

plt.scatter(spikes[:,0], spikes[:,1])
plt.xlabel("Time [s]")
plt.ylabel("Neuron ID")
plt.show()
    


