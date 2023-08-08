import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/calc_metrics')

# --------------------------------------------------------------------------------------- Precipiration metrics ----------------------------------------------------------------------------------------------------- #
run = False
if run:
    import pr as pM
    pM.run_pr_metrics(switch = {
            # choose data to calculate metric on
            'constructed_fields': False, 
            'sample_data':        False,

            # choose metrics to calculate
            'snapshot_pr':                    False,
            'rxday_sMean':                    True, 
            'rxday_tMean':                    True, 
            'percentiles':                    True, 
            'snapshot_percentiles':           False, 
            'meanInPercentiles':              True, 
            'meanInPercentiles_fixedArea':    True,
            'F_pr10':                         True,
            'snapshot_F_pr10':                False,
            'o_pr':                           True,
            
            # savve
            'save':               True
            }
        )
    
# -------------------------------------------------------------------------------------- Organization metrics ----------------------------------------------------------------------------------------------------- #
run = False
if run:
    import org_met as oM
    oM.run_org_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

        # choose metrics to calculate
        'rome':               True, 
        'rome_n':             True, 
        'ni':                 True, 
        'o_area':             True,
        'snapshot_obj':       False,
        
        # threshold
        'fixed_area':         False,

        # save
        'save':               True
        }
    )

        
# ------------------------------------------------------------------------------------- Large-scale environmental state ----------------------------------------------------------------------------------------------------- #
# -----------------------
# relative humidity (hur)
# -----------------------
run = False
if run:
    import large_scale_state.hur as hM 
    hM.run_hur_metrics(switch = {
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
    

# ----------------------------------------
# Outgoing Longwave Radiation (OLR) (rlut)
# ----------------------------------------
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
    

# ----------------------------------------
# Surface temperature (tas)
# ----------------------------------------
run = True
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
    

# ----------------------------------------
# Vertical pressure velocity (wap)
# ----------------------------------------
run = False
if run:
    import large_scale_state.wap as wM
    wM.run_wap_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

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


# ------------------------------------------------------------------------------------------------- clouds ----------------------------------------------------------------------------------------------------- #
run = False
if run:
    import clouds.cl as cM
    cM.run_cl_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

        # choose metrics to calculate
        'snapshot':           True, 
        'sMean':              True, 
        'tMean':              True, 

        # choose type of cloud
        'low_clouds':         True,
        'high_clouds':        False,

        # mask by
        'ascent':             False,
        'descent':            False,

        # save
        'save':               True
        }
    )










































