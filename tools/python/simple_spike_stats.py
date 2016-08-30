#!/usr/bin/python
import numpy as np
import pylab as pl
from auryntools import *
from auryntools.stats import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with default paramters. 
# This generates spk output files under /tmp/

filename  =  "/tmp/coba.0.e.spk"

sf = AurynBinarySpikeFile(filename)
spikes = sf.get_spikes()


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


