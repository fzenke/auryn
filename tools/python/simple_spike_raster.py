#!/usr/bin/python
import numpy as np
import pylab as pl
from auryntools import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with default paramters. 
# This generates spk output files under /tmp/


filename  =  "/tmp/coba.0.e.spk"
seconds = 0.1

sf = AurynBinarySpikeFile(filename)
spikes = np.array(sf.get_last(seconds))

pl.scatter(spikes[:,0], spikes[:,1])
pl.xlabel("Time [s]")
pl.ylabel("Neuron ID")
pl.show()
    


