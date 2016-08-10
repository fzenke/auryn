#!/usr/bin/python
import numpy as np
import pylab as pl
import struct


class AurynBinarySpikeFile:
    def __init__(self, filename, debug_output=False):
        # These params might have to adapted ot different Auryn datatypes and versions
        self.data_format = "@II"
        self.class_version = (0,8,0)

        self.debug = debug_output

        self.filename = filename
        self.frame_size = struct.calcsize(self.data_format)
        self.datafile = open(filename, "rb")

        # Reader spk file header
        data = self.datafile.read(self.frame_size)
        self.header = struct.unpack(self.data_format, data)
        self.timestep = 1.0/self.header[0]
        version_code = self.header[1]%1000
        self.file_version = (version_code%10, (version_code%100)/10, (version_code%1000)/100)
        if self.class_version != self.file_version:
            print("Warning! Version mismatch between the decoding tool and the file version.")
            print("AurynBinarySpikeFile %s"%str(self.class_version))
            print("Fileversion %s"%str(self.file_version))

        # Determine size of file
        self.datafile.seek(0,2)
        self.filesize = self.datafile.tell()
        self.num_frames = self.filesize/self.frame_size-1
        self.datafile.seek(self.frame_size,0)
        if self.debug:
            print "Filesize", self.filesize
            print "Number of frames", self.num_frames

    def __del__(self):
        self.datafile.close()

    def unpack(self, data):
        return struct.unpack(self.data_format, data)

    def get_frame(self,idx):
        '''
        Returns decoded contents of a frame with index idx as tuple 
        '''
        pos = (idx+1)*self.frame_size
        self.datafile.seek(pos,0)
        data = self.datafile.read(self.frame_size)
        frame = self.unpack(data)
        return frame

    def find_frame(self, time=0, lower=False):
        '''
        Find frame index of given time by bisection.
        '''
        idx_lo = 1
        idx_hi = self.num_frames
        while idx_lo+1<idx_hi:
            pivot = (idx_lo+idx_hi)/2
            at, nid = self.get_frame(pivot)
            ft = at*self.timestep
            if ft>time:
                idx_hi = pivot
            else:
                idx_lo = pivot
        if lower:
            return idx_lo
        else:
            return idx_hi


    def read_frames(self, bufsize):
        return self.datafile.read(bufsize*self.frame_size)


    def get_spikes_from_interval( self, t_start, t_stop ):
        idx_start = self.find_frame( t_start, lower=False )
        idx_stop = self.find_frame( t_stop, lower=True )
        start_pos = idx_start*self.frame_size
        num_elements = idx_stop-idx_start

        self.datafile.seek(start_pos,0)
        data = self.datafile.read(num_elements*self.frame_size)

        spikes = []
        for i in xrange(num_elements):
            at, nid = struct.unpack_from(self.data_format, data, i*self.frame_size)
            spikes.append((self.timestep*at, nid))
        return spikes

    def get_spike_times_from_interval( self, neuron_id=0, t_start=0, t_stop=1e32 ):
        idx_start = self.find_frame( t_start, lower=False )
        idx_stop = self.find_frame( t_stop, lower=True )

        idx_cur = idx_start
        start_pos = idx_start*self.frame_size
        self.datafile.seek(start_pos,0)

        bufsize = 1024*1024
        spike_times = []
        while idx_cur < idx_stop:
            if idx_stop-idx_cur<bufsize:
                bufsize = idx_stop-idx_cur
            idx_cur = idx_cur+bufsize
            data = self.read_frames(bufsize)
            for i in xrange(bufsize):
                at, nid = struct.unpack_from(self.data_format, data, i*self.frame_size)
                if nid==neuron_id:
                    spike_times.append(self.timestep*at)
        return spike_times

class AurynBinarySpikes:
    def __init__(self, filenames):
        self.filenames = filenames
        self.spike_files = []
        for filename in self.filenames:
            self.spike_files.append( AurynBinarySpikeFile(filename) )

    def get_spikes_from_interval( self, t_start, t_stop ):
        spikes = []
        for spk in self.spike_files:
            spikes.extend( spk.get_spikes_from_interval( t_start, t_stop ) )

        spikes.sort(key=lambda tup: tup[0])
        return spikes

    def compute_linear_receptive_field(self, stim_times, time_window=100e-3, max_neuron_id=1024):
        hist = np.zeros(max_neuron_id) 
        for t_spike in stim_times:
            spikes = self.get_spikes_from_interval(t_spike-time_window, t_spike)
            sar = np.array(spikes)[:,1]
            for spk in sar:
                if spk < max_neuron_id:
                    hist[spk] += 1
        return hist

    

# spikes = get_spikes_from_interval(200.0, 220.0)
# sar = np.array(spikes)
# pl.scatter(sar[:,0],sar[:,1])

t_win = 100e-3
num_neurons = 4096 
nid = 12

filename = "/home/zenke/data/sim/rf2.0.e.spk"
spkf     = AurynBinarySpikeFile(filename)

print "Getting firing times"
t_spikes = spkf.get_spike_times_from_interval(nid, 1000, 1200)

filenames = ["/home/zenke/data/sim/rf2.0.s.spk",    
             "/home/zenke/data/sim/rf2.1.s.spk",
             "/home/zenke/data/sim/rf2.2.s.spk",
             "/home/zenke/data/sim/rf2.3.s.spk" ]
spks      = AurynBinarySpikes(filenames)


print "Computing RF"
hist = spks.compute_linear_receptive_field(t_spikes, t_win, num_neurons)
pl.imshow(hist.reshape((64,64)), origin='lower')
pl.colorbar()
pl.show()
