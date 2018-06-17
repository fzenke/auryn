#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from auryntools import *
from auryntools.stats import *

# This code snipped assumes that you have run the example simulation
# sim_coba_binmon with default paramters. 
# This generates spk output files under /tmp/

filename  =  "/tmp/coba.0.e.spk"

sf = AurynBinarySpikeFile(filename)
spikes = sf.get_spikes()


plt.figure(figsize=(10,3))
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
plt.subplot(131)
plt.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
rate_hist(spikes, color="#00548c", bins=30)
plt.xlabel("Rate [Hz]")

plt.subplot(132)
plt.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
isi_hist(spikes, color="#ffcc00", bins=np.logspace(np.log10(1e-3), np.log10(10.0), 30))
plt.gca().set_xscale("log")
plt.xlabel("ISI [s]")

plt.subplot(133)
plt.tick_params( axis='y', which='both', left='off', right='off', labelleft='off') 
cvisi_hist(spikes, color="#c4000a", bins=30)
plt.xlabel("CVISI [1]")
plt.show()


