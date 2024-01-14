#!/sw/spack-levante/mambaforge-4.11.0-0-Linux-x86_64-sobz6z/bin/python
#SBATCH --job-name=find_files      # Specify job name
#SBATCH --account=bb1153
#SBATCH --partition=shared         # Specify partition name
#SBATCH --mem=14G
#SBATCH --time=24:00:00            # Set a limit on the total run time
#SBATCH --error=find_files.log.%j
#SBATCH --output=find_files.log.%j

import argparse


def get_from_cat(catalog, field, searchdict=None):
    """Call this to get all values of a field in the catalog as a sorted list"""
    if searchdict is not None and len(searchdict) > 0:
        cat = catalog.search(**searchdict)
    else:
        cat = catalog
    return sorted(cat.unique(field)[field]["values"])


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.description = """List files for a given variable and simulation."""
    parser.epilog = """Note that regular expressions can be used, but need to follow the "full" regex-syntax.
    "20-02" will search for exactly "2020-02" (no regex used).
    "20-02*" will search for anything *containing* "20-0" and an arbitrary number of "2"s following that, so "2020-03" also matches.
    "20-02.*" will search for anything *containing* "20-02" .
    "^20-02.*" will search for anything *starting with* "20-02" .
    "2020-.*-03T" will search for anything *containing*  "2020", followed by an arbitrary number of characters followed by "03T".
    
Use "" to leave variable_id or simulation_id empty.

Use 
find_files "" "" -f "experiment_id,source_id" 
to get a list of all experiments and participating models available
    """
    required_search_args = ("variable_id", "simulation_id")
    [parser.add_argument(x) for x in required_search_args]
    # optional search arguments (those with default are added below)
    optional_search_args = (
        "project",
        "institution_id",
        "source_id",
        "experiment_id",
        "realm",
        "frequency",
        "time_reduction",
        "grid_label",
        "level_type",
        "time_min",
        "time_max",
        "grid_id",
        "format",
        "uri",
    )
    for x in optional_search_args:
        parser.add_argument(f"--{x}", action="append")

    parser.add_argument(
        "-c", "--catalog_file", default="/work/ka1081/Catalogs/dyamond-nextgems.json"
    )
    parser.add_argument(
        "-f",
        "--print_format",
        default="uri",
        help="Comma separated list of columns to be plotted. e.g. 'variable_id,source_id'",
    )
    parser.add_argument(
        "--full", action="store_true", help="Print full dataset information"
    )
    parser.add_argument("--get", action="store_true", help="Get datasets from tapes")
    parser.add_argument("--datasets", action="store_true", help="List separate datasets")

    parser.add_argument(
        "--time_range",
        nargs=2,
        help="Find all files that contain data from a given range START END. \n--time_range 2020-02-01 2020-02-03 will give you all data for 2020-02-01 and 2020-02-02. \nNote that 2020-02-03 is smaller than any timestamp on 2020-02-03 in the string comparison logic.", 
    )

    pruned_dict = {k: v for k, v in vars(parser.parse_args()).items() if v is not None}
    search_args = {
        k: v
        for k, v in pruned_dict.items()
        if k in optional_search_args + required_search_args
    }
    for k in list(search_args.keys()):
        v = search_args[k]
        if len(v) == 1:
            search_args[k] = v[0]
        if len(v) == 0:
            del search_args[k]

    misc_args = {k: v for k, v in pruned_dict.items() if k not in search_args.keys()}
    return search_args, misc_args


def search_cat(cat, search_args, misc_args):
    hitlist = cat
    if len(search_args):
        hitlist = hitlist.search(**search_args)
    if "time_range" in misc_args.keys():
        tr = misc_args["time_range"]
        hitlist.df = hitlist.df[
            (hitlist.df["time_min"] <= tr[1]) * (hitlist.df["time_max"] >= tr[0])
        ]
    return hitlist


def print_results(hitlist, search_args, misc_args):
    if misc_args.get("full", False):
        import pandas as pd

        pd.set_option("display.max_columns", None)
        pd.set_option("max_colwidth", None)
        pd.set_option("display.width", 10000)
        print(hitlist.df)
    elif misc_args.get("datasets", False):
        import pandas as pd
        cols=[ x for x in hitlist.df if not x in ('uri','time_min','time_max') ]
        hitlist = (
            hitlist.df[cols]
            .drop_duplicates()
            .sort_values(cols)
            .to_string(index=False)
        )
        pd.set_option("display.max_columns", None)
        pd.set_option("max_colwidth", None)
        pd.set_option("display.width", 10000)
        print(hitlist)
    else:
        fmt = misc_args.get("print_format")
        cols = fmt.split(",")
        if len(cols) == 1:
            matches = get_from_cat(hitlist, fmt)
            [print(x) for x in matches]
        else:
            import pandas as pd

            pd.set_option("display.max_columns", None)
            pd.set_option("max_colwidth", None)
            pd.set_option("display.width", 10000)
            pd.set_option("display.max_rows", None)
            hitlist = (
                hitlist.df[cols]
                .drop_duplicates()
                .sort_values(cols)
                .to_string(index=False)
            )
            print(hitlist)


if __name__ == "__main__":
    search_args, misc_args = parse_args()

    import intake

    catalog_file = misc_args["catalog_file"]
    cat = intake.open_esm_datastore(catalog_file)

    hitlist = search_cat(cat, search_args, misc_args)
    try:
        import outtake
    except Exception as e:
        import sys
        print ("Warning: Failed to import outtake. Reason:", file=sys.stderr)
        print (e, file=sys.stderr)
        outtake = False
    if misc_args.get("get", False):
        if not outtake:
            import sys
            print(
                "Could not import outtake. No download support without it.",
                file=sys.stderr,
            )
            exit(1)
        cat = outtake.get(hitlist)

    cat._df["uri"] = cat._df["uri"].str.replace("file:///", "/")

    try:
        print_results(hitlist, search_args, misc_args)
    except ValueError:
        import sys

        print(
            "\nERROR: Could not find any matches for your query ",
            search_args,
            misc_args,
            "in catalog ",
            catalog_file,
            file=sys.stderr,
        )
        sys.exit(1)
