



def get_stability(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # Differnece in potential temperature between two vertical sections 
    # Temperature at pressure levels (K) 
    # Where there are no temperature values, exclude the associated pressure levels from the weights'''
    da = vB.load_variable({'ta': True}, switch, dataset, experiment, resolution, timescale)                    
    theta =  da * (1000e2 / da['plev'])**(287/1005) 
    plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
    da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
    w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']                 
    da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))
    return da


