


def get_netlw(switch, var_name, dataset, experiment, resolution, timescale):
    rlds, rlus, rlut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rlds', 'rlus', 'rlut']]
    da = -rlds + rlus - rlut
    return da

def get_netsw(switch, var_name, dataset, experiment, resolution, timescale):
    rsdt, rsds, rsus, rsut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rsdt', 'rsds', 'rsus', 'rsut']]
    da = rsdt - rsds + rsus - rsut
    return da




