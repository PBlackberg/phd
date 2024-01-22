# ---------------------------------------------------------------------------------- Easy-gems helper functions --------------------------------------------------------------------------------------------------- #
def fix_time_axis(data):
    """Turn icon's yyyymmdd.f time axis into actual datetime format.

    This will fail for extreme values, but should be fine for a few centuries around today.
    """
    if (data.time.dtype != "datetime64[ns]") and (
        data["time"].units == "day as %Y%m%d.%f"
    ):
        data["time"] = pd.to_datetime(
            ["%8i" % x for x in data.time], format="%Y%m%d"
        ) + pd.to_timedelta([x % 1 for x in data.time.values], unit="d")


def get_from_cat(catalog, columns):
    """A helper function for inspecting an intake catalog.

    Call with the catalog to be inspected and a list of columns of interest."""
    import pandas as pd

    pd.set_option("max_colwidth", None)  # makes the tables render better

    if type(columns) == type(""):
        columns = [columns]
    return (
        catalog.df[columns]
        .drop_duplicates()
        .sort_values(columns)
        .reset_index(drop=True)
    )

def get_list_from_cat(catalog, column):
    """A helper function for getting the contents of a column in an intake catalog.

    Call with the catalog to be inspected and the column of interest."""
    return sorted(catalog.unique(column)[column]["values"])

def make_tempdir(name):
    """Creates a temporary directory in your /scratch/ and returns its path as string"""

    uid = getuser()
    temppath = os.path.join("/scratch/", uid[0], uid, name)
    os.makedirs(temppath, exist_ok=True)
    return temppath

def find_grids(dataset):
    """Generic ICON Grid locator

    This function checks an xarray dataset for attributes that contain "grid_file_uri", and checks if it can map them to a local path.
    It also checks for "grid_file_name"

    It returns a list of paths on disk that are readable (os.access(x, os.R_OK)).
    """
    uris = [
        dataset.attrs[x] for x in dataset.attrs if "grid_file_uri" in x
    ]  # this thing might come in one of various names...
    search_paths = [
        re.sub("http://icon-downloads.mpimet.mpg.de", "/pool/data/ICON", x)
        for x in uris
    ] + [
        os.path.basename(x) for x in uris
    ]  # plausible mappings on mistral.
    if "grid_file_path" in dataset.attrs:
        search_paths.append(dataset.attrs["grid_file_path"])
        search_paths.append(
            os.path.basename(dataset.attrs["grid_file_path"])
        )  # also check the current dir.
    paths = [
        x for x in search_paths if (os.access(x, os.R_OK))
    ]  # remove things that don't exist.
    if not paths:
        message = "Could not determine grid file!"
        if search_paths:
            message = message + "\nI looked in\n" + "\n".join(search_paths)
        if uris:
            message = message + (
                "\nPlease check %s for a possible grid file" % (" or ").join(uris)
            )
        raise Exception(message)
    if len(set(paths)) > 1:
        print(
            "Found multiple conflicting grid files. Using the first one.",
            file=sys.stderr,
        )
        print("Files found:", file=sys.stderr)
        print("\n".join(paths), file=sys.stderr)
    return paths

def add_grid(dataset):
    """Generic icon grid adder.

    Calls find_grids to locate a grid file, and - if it finds one - adds this grid file to a Dataset.

    also tries to ensure that clon has the same dimensions as the data variables.
    """
    paths = find_grids(dataset)
    grid = xr.open_dataset(paths[0])
    rename = (
        {}
    )  # icon uses different dimension names in the output than in the grid file. (whyever...)
    if "ncells" in dataset.dims:
        grid_ncells = grid.clon.dims[0]
        rename = {grid_ncells: "ncells"}
    drops = set(
        [x for x in grid.coords if x in dataset.data_vars or x in dataset.coords]
        + [x for x in grid.data_vars if x in dataset.data_vars or x in dataset.coords]
    )
    return xr.merge((dataset.drop(drops), grid.rename(rename)))




