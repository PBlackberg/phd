import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/plot')

# -------------------------------------------------------------------------------------------- Map plot ----------------------------------------------------------------------------------------------------- #
plot = False
if plot:
    import map_plot as mP
    mP.run_map_plot(switch = {
        # metrics
        'pr':                  False,
        'pr97':                False,
        'pr99':                False,
        'rx1day_pr':           False,
        'rx5day_pr':           False,

        'obj':                 False,

        'hur':                 False,
        'wap':                 False,
        'tas':                 False,

        'lcf':                 False,
        'hcf':                 False,

        'rlut':                False,
        'hus':                 False,


        # metric calculation
        'snapshot':            True,
        'climatology':         False,
        'change with warming': False,


        # masked by
        'descent':             False,
        'ascent':              False,
        'fixed_area':          False,
        'per_kelvin':          False,
        

        # show/save
        'one dataset':         False,
        'show':                True,
        'save':                True,
        'save to desktop':     False
        }
    )


# ----------------------------------------------------------------------------------------- intra-model scatter ----------------------------------------------------------------------------------------------------- #
plot = False
if plot:
    import scatter_plot as sP
    sP.run_scatter_plot(switch = {
        # metrics
            # organization
            'rome':                False,
            'areafraction':        True,
            'ni':                  True,

            # other
            'pr':                  False,
            'pr97':                False,
            'pr99':                False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'hur':                 False,
            'wap':                 False,
            'tas':                 False,

            'lcf':                 False,
            'hcf':                 False,

            'rlut':                False,
            'hus':                 False,

        # organization masked by
        'fixed_area':          False,

        # other variable masked by
        'descent':             False,
        'ascent':              False,

        # metric calculation
        'anomalies':           True,

        # plot modifications
        'bins':                True,
        'xy':                  True,

        # run/show/save
        'one dataset':         False,
        'show':                True,
        'save':                True,
        'save to desktop':     False
        }
    )



# ----------------------------------------------------------------------------------------- inter-model scatter ----------------------------------------------------------------------------------------------------- #
plot = True
if plot:
    import scatter2_plot as s2P    
    s2P.run_scatter_plot(switch = {
        # metrics
            # organization
            'rome':                False,
            'ni':                  False,
            'areafraction':        False,

            # other
            'pr':                  False,
            'pr99':                False,
            'pr99_sMean':          False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'hur':                 True,
            'wap':                 False,
            'tas':                 False,
            'ecs':                 False,

            'lcf':                 False,
            'hcf':                 False,

            'rlut':                True,
            'hus':                 False,

        # masked by
        'descent':             False,
        'ascent':              False,
        'fixed_area':          False,
        'per_kelvin':          False,

        # metric calculation
        'climatology':         True,
        'change with warming': False,

        # plot modifications
        'xy':                  True,

        # show/save
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }
    )









