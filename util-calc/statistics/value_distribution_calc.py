

import xarray as xr
import numpy as np


def get_percentile_values(aList, dimensions, percentiles = np.arange(1, 101, 1)):
    return aList.quantile(percentiles/100, dim=dimensions).compute()

def calc_freq_occur(aList, ymin = 0, ymax = 30, nb_bins = 7):
    ''' In units of % '''
    edges = np.linspace(ymin, ymax, nb_bins)
    freq_occur = []
    for i in range(len(edges)-1):
        if i == len(edges)-1:
            freq_occur_bin = (xr.where((aList>=edges[i]) & (aList<=edges[i+1]), 1, 0).sum() / len(aList)).data # include points right at the end of the distribution
        else:
            freq_occur_bin = (xr.where((aList>=edges[i]) & (aList<edges[i+1]), 1, 0).sum() / len(aList)).data
        freq_occur.append(freq_occur_bin)
    bins_middle = edges[0:-1] + (edges[1] - edges[0])/2
    return xr.DataArray(np.array(freq_occur)*100), bins_middle 
















