'''
# --------------------
#  SCP file transfer
# --------------------
'''



import os
import subprocess



def find_computer_from():
    computer_from = 'local'     if os.path.expanduser("~") == '/Users/cbla0002'     else None
    computer_from = 'gadi'      if os.path.expanduser("~") == '/home/565/cb4968'    else computer_from
    computer_from = 'levante'   if os.path.expanduser("~") == '/home/b/b382628'     else computer_from
    if computer_from == None:
        print('cannot identify computer_from')
    return computer_from

def find_computer_to(folder_to):
    computer_to = 'local'   if '/Users/cbla0002'        in folder_to    else None
    computer_to = 'gadi'    if '/scratch/w40/cb4968'    in folder_to    else computer_to
    computer_to = 'levante' if '/scratch/b/b382628'     in folder_to    else computer_to
    if computer_to == None:
        print('cannot identify computer_to')
    return computer_to

def create_cmd(folder_from, folder_to):
    computer_from = find_computer_from() 
    computer_to = find_computer_to(folder_to) 
    scp_command = ''
    if computer_from == 'local':    
        if computer_to == 'gadi':
            scp_command = ["scp", "-r", folder_from, f"cb4968@gadi-dm.nci.org.au:{folder_to}"]
        if computer_to == 'levante':
            scp_command = ["scp", "-r", folder_from, f"b382628@levante.dkrz.de:{folder_to}"]
    if computer_from == 'gadi':    
        if computer_to == 'local':
            scp_command = ["scp", "-r", f"cb4968@gadi-dm.nci.org.au:{folder_from}", folder_to]
    if computer_from == 'levante':    
        if computer_to == 'local':
            scp_command = ["scp", "-r", f"b382628@levante.dkrz.de:{folder_from}", folder_to]
    if scp_command == '':
        print('cannot create scp command')
    return scp_command

def scp_transfer(folder_from, folder_to):
    scp_command = create_cmd(folder_from, folder_to)
    try:
        subprocess.run(scp_command, check=True)
        print(f"Folder '{folder_from}' transferred successfully to '{folder_to}'")
    except subprocess.CalledProcessError as e:
        print(f"Error transferring folder: {e}")



# --------------------
#  Choose folders
# --------------------
if __name__ == '__main__':
    print('file transfer starting')

    folder_from = '/scratch/w40/cb4968/metrics/wap'

    folder_to = '/Users/cbla0002/Desktop/'

    print(f'transferring from {find_computer_from()} to {find_computer_to(folder_to)}')
    scp_transfer(folder_from, folder_to)


