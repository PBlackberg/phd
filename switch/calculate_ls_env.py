import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/calc_metrics')


# --------------------
# Surface temperature
# --------------------
run = False
if run:
    import large_scale_state.tas as tM
    tM.run_tas_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        False,

        # choose metrics to calculate
        'sMean':              True, 
        'tMean':              True, 
        'snapshot':           True, 
        
        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )

# ------------------
# relative humidity
# ------------------
run = True
if run:
    import large_scale_env.hur as hM 
    hM.run_hur_metrics(switch = {
        # choose data to calculate metric on
        'constructed fields': False, 
        'sample data':        False,
        'gadi data':          True,

        # choose metrics to calculate
        'sMean':              True, 
        'tMean':              True, 
        'snapshot':           True, 
        
        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )
    
# ----------------------------
#  Outgoing Lonwave Radiation
# ----------------------------
run = False
if run:
    import large_scale_state.rlut as rM 
    rM.run_rlut_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        False,

        # choose metrics to calculate
        'sMean':              True, 
        'tMean':              True, 
        'snapshot':           True, 

        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )
    
# ---------------------------
# Vertical pressure velocity
# ---------------------------
run = False
if run:
    import large_scale_state.wap as wM
    wM.run_wap_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        False,

        # choose metrics to calculate
        'sMean':              True, 
        'tMean':              True, 
        'snapshot':           True, 
        
        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )



