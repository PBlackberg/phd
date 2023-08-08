import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/plot')

# -------------------------------------------------------------------------------------------- Map plot ----------------------------------------------------------------------------------------------------- #
plot = True
if plot:
    import map_plot as mP
    mP.run_map_plot(switch = {
        # metrics
        'pr':                  False,
        'pr99':                False,
        'rx1day_pr':           False,
        'rx5day_pr':           False,

        'obj':                 False,

        'wap':                 False,
        'tas':                 False,

        'hus':                 False,
        'hur':                 True,
        'rlut':                False,

        'lcf':                 False,
        'hcf':                 False,


        # metric calculation
        'snapshot':            False,
        'climatology':         True,
        'change with warming': False,

        # masked by
        'descent':             False,
        'ascent':              False,
        'per_kelvin':          False,
        
        # show/save
        'one dataset':         True,
        'show':                True,
        'save':                False,
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
            'rome':                True,

            # other
            'pr':                  False,
            'pr99':                False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'wap':                 False,
            'tas':                 False,

            'hus':                 False,
            'hur':                 True,
            'rlut':                False,

            'lcf':                 False,
            'hcf':                 False,

        # masked by
        'descent':             False,
        'ascent':              False,

        # metric calculation
        'anomalies':           True,

        # plot modifications
        'bins':                True,
        'xy':                  True,

        # run/show/save
        'one dataset':         False,
        'run':                 False,
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }
    )

# ----------------------------------------------------------------------------------------- inter-model scatter ----------------------------------------------------------------------------------------------------- #
plot = False
if plot:
    import scatter2_plot as s2P    
    s2P.run_scatter_plot(switch = {
        # metrics
            # organization
            'rome':                True,

            # other
            'ecs':                 True,
            'pr':                  False,
            'pr99':                False,
            'pr99_meanIn':         False,
            'rx1day_pr':           False,
            'rx5day_pr':           False,

            'wap':                 False,
            'tas':                 False,

            'hus':                 False,
            'hur':                 False,
            'rlut':                False,

            'lcf':                 False,
            'hcf':                 False,

        # masked by
        'descent':             False,
        'ascent':              False,

        # metric calculation
        'climatology':         True,
        'change with warming': False,

        # plot modifications
        'xy':                  True,

        # show/save
        'run':                 True,
        'show':                True,
        'save':                False,
        'save to desktop':     False
        }
    )










































