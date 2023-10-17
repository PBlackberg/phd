import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/calc_metrics')

# --------------
# Precipitation
# --------------
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
    
# -------------
# Organization
# -------------
run = False
if run:
    import org_met as oM
    oM.run_org_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

        # visualization
        'obj_snapshot':       False,

        # metrics to calculate
        'rome':               True, 
        'rome_n':             True, 
        'ni':                 True, 
        'o_area':             True,

        # threshold
        'fixed_area':         True,

        # save
        'save':               True
        }
    )


