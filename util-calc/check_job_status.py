'''
# --------------------
#  Check job status
# --------------------
This script checks the status of active jobs
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import subprocess



# ---------------
#     Funcs
# ---------------
def get_active_jobs():
    """Get a dictionary of active jobs and their statuses from qstat."""
    try:
        output = subprocess.check_output(['qstat'], encoding='utf-8')
    except subprocess.CalledProcessError as e:
        print(f"Error fetching job status: {e}")
        return {}
    jobs = {}
    for line in output.splitlines()[2:]:  # Skip the header lines
        parts = line.split()
        if len(parts) >= 5:
            job_id = parts[0]
            status = parts[4]
            jobs[job_id] = status
    return jobs

def list_all_active_jobs():
    """Print all active jobs and their statuses."""
    active_jobs = get_active_jobs()
    if active_jobs:
        for job_id, status in active_jobs.items():
            print(f"Job {job_id} is {status}")
    else:
        print("No active jobs found.")



# ---------------
#     Run
# ---------------
if __name__ == "__main__":
    list_all_active_jobs()




