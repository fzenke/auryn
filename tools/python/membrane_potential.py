#!/usr/bin/python
import numpy as np
import pylab as pl
from auryntools import *

# This code snipped assumes that you have run the example simulation
# sim_epsp_binmon with default parameters and adjusted the below 
# filename to its output.

filename  =  "../../build/release/examples/out_epsp.0.bmem"
t_from=0.2
t_to  =2.5

sf = AurynBinaryStateFile(filename)
mem = np.array(sf.get_data(t_from, t_to))

pl.plot(mem[:,0], mem[:,1])
pl.xlabel("Time [s]")
pl.ylabel("Membrane potential [V]")
pl.show()
    


