#!/usr/bin/python
import numpy as np
import struct


filename = "/home/zenke/data/sim/rf3.0.e.spk"
datafile = open(filename, "rb")
data_format = "@II"
data_struct_size = struct.calcsize(data_format)

# Reader spk file header
data = datafile.read(100)
header = struct.unpack_from(data_format, data)
timestep = 1.0/header[0]
version_code = header[1]%1000
file_version = (version_code%10, (version_code%100)/10, (version_code%1000)/100)
print "File version",file_version
print "File data struct size",data_struct_size

datafile.seek(0,2)
filesize = datafile.tell()
num_frames = filesize/data_struct_size-1
datafile.seek(data_struct_size,0)
print "Filesize", filesize
print "Number of frames",num_frames


def get_frame(idx):
    '''
    Returns decoded contents of a frame with index idx as tuple 
    '''
    pos = (idx+1)*data_struct_size
    datafile.seek(pos,0)
    data = datafile.read(data_struct_size)
    frame = struct.unpack(data_format, data)
    return frame

def find_frame(time=0, lower=False):
    '''
    Find frame index of given time by bisection.
    '''
    idx_lo = 0
    idx_hi = num_frames
    while idx_lo+1<idx_hi:
        pivot = (idx_lo+idx_hi)/2
        at, nid = get_frame(pivot)
        ft = at*timestep
        if ft>time:
            idx_hi = pivot
        else:
            idx_lo = pivot
    if lower:
        return idx_lo
    else:
        return idx_hi

def get_spikes( t_start, t_stop ):
    idx_start = find_frame( t_start, lower=False )
    idx_stop = find_frame( t_stop, lower=True )
    start_pos = (idx_start+1)*data_struct_size
    datafile.seek(start_pos,0)
    num_elements = idx_stop-idx_start
    data = datafile.read(num_elements*data_struct_size)

    spikes = []
    for i in xrange(num_elements):
        at, nid = struct.unpack_from(data_format, data, i*data_struct_size)
        spikes.append((timestep*at, nid))
    return spikes


spikes = get_spikes(1.0, 2.0)
print np.array(spikes)
