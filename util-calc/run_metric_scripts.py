'''
# ------------------------
#   Run metric scripts
# ------------------------
Executes metric script in parallel (calculation of metric from each model is submitted as individual jobs)
Function:
    run_scripts_in_parallel()

Input:
    metric_script

Output:
    saves metric in files according to filestructure in util-data/metric_data
'''

        
# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import subprocess



# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD        # settings from util-core/choose_datasets.py         



# -------------------------
#  List available scripts
# -------------------------
# ------------------------------------------------------------------------------------ metric scripts --------------------------------------------------------------------------------------------------- #
python_script = {
    'conv_org_metrics': 'util-calc/conv_org/conv_org_metrics.py'
    }


# ------------------------------------------------------------------------------------ resource script --------------------------------------------------------------------------------------------------- #
pbs_script = {
    'conv_org_metrics': 'util-bash/gadi_job.pbs'
    }


# ---------------
#   Run scripts
# ---------------
def run_scripts_in_parallel(switch_script):
    for metric_type in [k for k, v in switch_script.items() if v]:
        print(f'Running {metric_type}')
        for dataset in cD.datasets:

            command = ["qsub", "-v", f"PYTHON_SCRIPT={python_script[metric_type]},MODEL_IDENTIFIER={dataset}", pbs_script[metric_type]]
            subprocess.run(command, check=True)

            print(f'\t\t submitted script: {python_script[metric_type]}')
            print(f'\t\t with resources from: {pbs_script[metric_type]}')
            print(f'\t\t for dataset: {dataset}')



if __name__ == '__main__':
    switch_script = {
        'conv_org_metrics':     True
        }

    run_scripts_in_parallel(switch_script)




