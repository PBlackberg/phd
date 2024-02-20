''' 
# --------------------
#   get_data - Intake
# --------------------
This script gets data from DYAMOND and NextGEMS simulations, using the intake module

'''

# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import intake
import outtake
    


# --------------------
#    get catalogue
# --------------------
def get_catalogue():
    catalog_file = "/work/ka1081/Catalogs/dyamond-nextgems.json"
    cat = intake.open_esm_datastore(catalog_file)
    return cat

def print_available_datasets():
    cat = get_catalogue()
    columns = ["project", "experiment_id", "source_id", "simulation_id"]
    result = (cat.df[columns].drop_duplicates().sort_values(columns).reset_index(drop=True))
    print(result)
    
def print_dataset_info(simulation_id, columns = ''):
    cat = get_catalogue()
    df = cat.search(simulation_id= simulation_id).df
    if not columns:
        print(f'Available columns in {simulation_id} dataframe:')
        for column in df.columns.tolist():
            print(column)
    else:
        print(f'{simulation_id} search:')
        print(df[columns])
    return df

def get_files_intake(simulation_id, frequency, variable_id, realm = 'atm'):
    cat = get_catalogue()
    hits = cat.search(simulation_id = simulation_id, realm = realm, frequency = frequency, variable_id = variable_id)
    files = hits.df['uri'].tolist()
    return files

def get_data_intake(simulation_id, variable_id, frequency, realm = 'atm', time_chunk = 1):
    cat = get_catalogue()
    hits = cat.search(simulation_id = simulation_id, realm = 'atm', variable_id = variable_id, frequency = frequency)
    dataset_dict = hits.to_dataset_dict(cdf_kwargs={"chunks": {"time": time_chunk}})
    keys = list(dataset_dict.keys())
    ds = dataset_dict[keys[0]]
    da = ds[variable_id]
    return da



if __name__ == '__main__':
    import pandas as pd
    import xarray as xr
    pd.set_option("max_colwidth", None)  # makes the tables render better

    # Check data catalogue
    check_it = False
    if check_it:
        print_available_datasets()
        df = print_dataset_info(simulation_id = 'ngc2013')
        columns = ["realm", "variable_id", 'frequency']
        df = print_dataset_info(simulation_id = 'ngc2013', columns = columns)
        print(df['frequency'].unique())

    # Check data from file list
    get_it = False
    if get_it:
        simulation_id, frequency, variable_id  = 'ngc2013', '3hour', 'pr' # can also do realm
        files = get_files_intake(simulation_id, frequency, variable_id)
        [print(file) for file in files[0:2]]
        [print(file) for file in files[-2::]]
        ds = xr.open_dataset(files[0])
        print(ds)
        # print("Variables in dataset:")
        # for var in list(ds.data_vars.keys()):
        #     print(var)
        # print(ds.time)

    # get data as full dataset (takes longer)
    get_data = False
    if get_data:
        simulation_id, frequency, variable_id  = 'ngc2013', '3hour', 'pr' # can also do realm
        da = get_data_intake(simulation_id, variable_id, frequency)
        print(da)




