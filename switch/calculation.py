import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/calc_metrics')

# -------------------------------------------------------------------------------------- Organization metrics ----------------------------------------------------------------------------------------------------- #
import org_met as oM
oM.run_org_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        True,
    'gadi_data':          False,

    # choose metrics to calculate
    'obj_snapshot':       True,
    'rome':               False, 
    'rome_n':             False, 
    'ni':                 True, 
    'o_area':             False,
    
    # run/savve
    'run':                True,
    'save':               False
    }
)

# --------------------------------------------------------------------------------------- Precipiration metrics ----------------------------------------------------------------------------------------------------- #
import pr as pM
pM.run_pr_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,

    # choose metrics to calculate
    'snapshot':           False,
    'rxday_pr':           False, 
    'percentiles':        False, 
    'meanInPercentiles':  False, 
    'F_pr10':             False,
    'o_pr':               False,

    # run/savve
    'run':                False,
    'save':               False
    }
)

# ------------------------------------------------------------------------------------- Large-scale environmental state ----------------------------------------------------------------------------------------------------- #
# -----------------------
# relative humidity (hur)
# -----------------------
import large_scale_state.hur as hM 
hM.run_hur_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,

    # choose metrics to calculate
    'snapshot':           False, 
    'sMean':              False, 
    'tMean':              False, 

    # mask data by
    'ascent':             False,
    'descent':            False,
    
    # run/save
    'run':                False,
    'save':               False
    }
)

# ----------------------------------------
# Outgoing Longwave Radiation (OLR) (rlut)
# ----------------------------------------
import large_scale_state.rlut as rM 
rM.run_rlut_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,
    
    # choose metrics to calculate
    'snapshot':           False, 
    'sMean':              False, 
    'tMean':              False, 

    # mask data by
    'ascent':             False,
    'descent':            False,

    # run/save
    'run':                False,    
    'save':               False
    }
)

# ----------------------------------------
# Surface temperature (tas)
# ----------------------------------------
import large_scale_state.tas as tM
tM.run_tas_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,
    
    # choose metrics to calculate
    'snapshot':           False, 
    'sMean':              False, 
    'tMean':              False, 

    # mask data by
    'ascent':             False,
    'descent':            False,

    # run/save
    'run':                False,    
    'save':               False
    }
)

# ----------------------------------------
# Vertical pressure velocity (wap)
# ----------------------------------------
import large_scale_state.wap as wM
wM.run_wap_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,
    
    # choose metrics to calculate
    'snapshot':           False, 
    'sMean':              False, 
    'tMean':              False, 

    # mask data by
    'ascent':             False,
    'descent':            False,

    # run/save
    'run':                False,    
    'save':               False
    }
)

# ------------------------------------------------------------------------------------------------- clouds ----------------------------------------------------------------------------------------------------- #
import clouds.cl as cM
cM.run_cl_metrics(switch = {
    # choose data to calculate metric on
    'constructed_fields': False, 
    'sample_data':        False,
    'gadi_data':          False,

    # choose type of cloud
    'low_clouds':         False,
    'high_clouds':        False,

    # choose metrics to calculate
    'snapshot':           False, 
    'sMean':              False, 
    'tMean':              False, 

    # mask data by
    'ascent':             False,
    'descent':            False,

    # run/save
    'run':                False,   
    'save':               False
    }
)










































