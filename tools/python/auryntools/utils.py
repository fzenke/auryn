
def select_spikes(spikes,t_start,t_stop):
    """ Selects temporal subset of spikes from spike array.

    Parameters:

        spikes : A 2D array of spikes ( spiketime x neuron id ). This array needs to be sorted by time.
        t_start : The starting time point to select
        t_stop : The end time point to select

    returns:
        A 2D array of spikes 
    """
    ts = spikes[:,0]
    start = np.searchsorted(ts,t_start)
    stop  = np.searchsorted(ts,t_stop, side='right')
    return spikes[start:stop]

def raster_plot(spikes,t_start,t_stop):
    """ Plots a spike raster plot from a spike array
    
    Parameters:
        
        spikes : An array with spike times
        t_start : starting time to plot
        t_stop : last time to plot
    """
    d = select_spikes(spikes,t_start,t_stop)
    plt.scatter(d[:,0], d[:,1], marker='.', s=1.0)
