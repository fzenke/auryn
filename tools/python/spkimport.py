#!/usr/bin/python
import numpy as np
import pylab as pl
import struct


class AurynBinarySpikeFile:
    '''
    This class gives abstract access to binary Auryn spike raster file (spk).

    Public methods:
    get_spikes: extracts spikes (tuples of time and neuron id) for a given temporal range.
    get_spike_times_from_interval: extracts the spike times of a single unit and a given temporal range.
    '''
    def __init__(self, filename, debug_output=False):
        # These params might have to adapted ot different Auryn datatypes and versions
        self.data_format = "@II"
        self.class_version = (0,8,0)

        self.debug = debug_output

        self.filename = filename
        self.frame_size = struct.calcsize(self.data_format)
        self.datafile = open(filename, "rb")
        # TODO add exception handling

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


    def get_spikes( self, t_start=0.0, t_stop=1e32, max_id=1e32 ):
        idx_start = self.find_frame( t_start, lower=False )
        idx_stop = self.find_frame( t_stop, lower=True )
        start_pos = idx_start*self.frame_size
        num_elements = idx_stop-idx_start

        self.datafile.seek(start_pos,0)
        data = self.datafile.read(num_elements*self.frame_size)

        spikes = []
        for i in xrange(num_elements):
            at, nid = struct.unpack_from(self.data_format, data, i*self.frame_size)
            if nid<max_id:
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
    '''
    A wrapper class for easy extraction of spikes from multiple spk files from different ranks.
    '''
    def __init__(self, filenames):
        self.filenames = filenames
        self.spike_files = []
        for filename in self.filenames:
            self.spike_files.append( AurynBinarySpikeFile(filename) )

    def get_spikes( self, t_start=0.0, t_stop=1e32, max_id=1e32 ):
        spikes = []
        for spk in self.spike_files:
            spikes.extend( spk.get_spikes( t_start, t_stop, max_id ) )

        spikes.sort(key=lambda tup: tup[0])
        return spikes

    def time_triggered_histogram(self, trigger_times, time_window=100e-3, max_neuron_id=1024):
        '''
        Sums spikes within a given time window which precede given trigger times.

        This function can be used to compute reverese correlations, for instance to
        compute a linear receptive field. Note that for that reason the given time 
        window precedes the trigger time.

        Keyword arguments:
        trigger_times -- list of trigger times
        time_window -- size of time window to sum over in seconds (default 0.1s)
        max_neuron_id -- the number of neurons
        ''' 
        hist = np.zeros(max_neuron_id) 
        for t_spike in trigger_times:
            spikes = self.get_spikes(t_spike-time_window, t_spike)
            sar = np.array(spikes, dtype=int)[:,1]
            counts = np.bincount(sar, minlength=max_neuron_id)
            hist += counts
        return hist


def main():
    # running the example program sim_coba_binmon will
    # generate this file
    filenames = ["/tmp/coba.0.e.spk"]

    t_start = 0.0
    t_end   = 0.3
    n_max = 200

    spikecontainer = AurynBinarySpikes(filenames)
    spikes = np.array(spikecontainer.get_spikes(t_start,t_end,max_id=n_max))

    pl.scatter(spikes[:,0], spikes[:,1])
    pl.xlabel("Time [s]")
    pl.ylabel("Neuron ID")
    pl.show()
    

if __name__ == "__main__":
    main()

