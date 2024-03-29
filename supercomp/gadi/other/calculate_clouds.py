import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/calc_metrics')

# -----------------------
# Clouds - cloud fraction
# -----------------------
run = True
if run:
    import clouds.cl as cM
    cM.run_cl_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        False,
        'gadi_data':          True,

        # choose metrics to calculate
        'snapshot':           True, 
        'sMean':              True, 
        'tMean':              True, 

        # choose type of cloud
        'low_clouds':         True,
        'high_clouds':        False,

        # mask by
        'ascent':             False,
        'descent':            True,

        # save
        'save':               True
        }
    )








