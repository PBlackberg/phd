.. _for-beginners:

DYAMOND Winter dpp0029/dpp0033 - a beginner's guide
=========================================================

I. The output
---------------------------

1. What output is available?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The available variables and their abbreviations are stored in the `Table of variables <https://easy.gems.dkrz.de/_static/DYAMOND/WINTER/variable_table.html>`__ . The output is structured in files, see :ref:`dpp0029output` for the overview over the variables and the files in which they are stored (for the experiment dpp0029; the experiment dpp0033 has the same structure but shifted by a year). Each file corresponds to a day’s worth of observations, but the time periods over which the output is given can vary (this can also be found in the output description).

.. admonition:: Remarks

   Horizontally the output is stored in a one-dimensional array (‘ncells’ in the files). In order to be able to assess it depending on latitude and longitude one needs to attach a grid file in which latitudes and longitudes of the cells are stored. A land-sea mask is also stored in this grid file.

   Vertically the output is structured in levels (1 to 90), the lowest being 90. Geographic heights corresponding to these levels are also stored in a separate grid file. Note that they are absolute and depend on the topography. Thus, for accessing the heights over sea it is sufficient to attach the heights from one single grid point over sea to all the output.

   The time variable is stored as real numbers, e.g. the 6AM on 1 February 2020 is given as :math:`20200201.25`. For practical reasons it might be useful to change this into the datetime format.
   (See below for code to attach the grid and change the time.)



.. admonition:: Example

   I need the surface temperature output. The corresponding variable is :code:`ts`, and it is available in the files

   :code:`dpp0029_atm_2d_ml_<datetime>.nc`

   for January and February with 6h intervals,

   :code:`dpp0029_atm4_2d_ml_<datetime>.nc`

   for January and February with 15min intervals and

   :code:`dpp0029_atm_2d_ml_<datetime>.nc`

   starting from March with 1 day intervals (i.e. one observation per file corresponding to the daily mean).

2. How to access it and obtain more information?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One possibility is to log into Levante over the terminal by typing

:code:`ssh -X x123456@levante.dkrz.de`

and replacing :code:`x123456` by your DKRZ user ID. By typing

:code:`ls /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/*20200201*`

you obtain the list of files corresponding to a given day (here 01.02.2020) in the dpp0029 experiment. For details on a particular file, type :code:`ncdump -h` and the name of the file.

.. admonition:: Example

   To see what kind of output is stored in the files of the type :code:`dpp0029_atm_2d_ml_<datetime>.nc` write

   :code:`ncdump -h /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_2d_ml_20200201T000000Z.nc`

..
   Note: I can log into The GEMS Freva, but I cannot access the “ncdump”-information on particular files. The window prompts me to resubmit my password, but it is not recognised as correct.

..
   Note: Apparently the files I’ve been working with before (dpp0029 stored in Levante at /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/) are structured in a different way. Which data is meant to be used for NextGEMS? If I understood correctly, part of what Freva does is unifying the output, so maybe describing the current status of where the output is and how it is structured makes little sense?

II. Processing the output
---------------------------

CDO
~~~~~~~~~~~~~~~~~~~~~

The Climate Data Operators (CDO) is a collection of tools to group and perform basic operations on data. It is used to reduce the amount of data to whatever is necessary BEFORE plotting it or performing a statistical analysis.
Frequent uses include selecting parameters, spatial or temporal subsets of data, concatenating files, building averages/minima/maxima (and performing other basic math operations) as well as interpolating the data on a horizontal grid (this is called regridding).
This is often an efficient way to preprocess the data.

You can type CDO commands directly into the console after logging into Levante. However, for manipulating data you need to request some memory resources. This is done by typing a corresponding command into the terminal, for instance

:code:`salloc --x11 -p interactive -A xz0123 -n 1 --mem 8000 -t 240`

where :code:`xz0123` is replaced by your project name.

.. admonition:: Examples

   :code: `cdo -info /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_2d_ml_20200201T000000Z.nc`

   provides basic information about the contents of the file, including the maxima, minima and mean values of the variables.

   The command (read and executed from right to left)

   :code: `cdo -timmean -selname,ts,pr -cat /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_2d_ml_20200121*.nc /work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_2d_ml_20200122*.nc Outfile.nc`

   concatenates files from January, 21 and 22, selects the variables of surface temperature and precipitation, calculates the respective time means and writes the result in a new file called :code:`Outfile.nc`

   :code: `cdo -sellonlatbox,-56.5,-59,12,14.5 -remap,griddes01x01.txt,weight_dpp0029_01x01.nc -seltimestep,1/40 -selname,va -cat '/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_3d_v_ml_2020*.nc' Outfile.nc`

   is a CDO command that concatenates files of the form :code:`/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/dpp0029_atm_3d_v_ml_2020*.nc` , selects the variable :code:`va`, selects the first 40 time points, interpolates the obtained data on a fine grid (the grid files are created beforehand following the instructions in `MPI Wiki Regridding <https://wiki.mpimet.mpg.de/doku.php?id=analysis:postprocessing_icon:regridding:shell:start>`__) and finally cuts out a longitude-latitude box. The resulting output is then saved as :code:`Outfile.nc` in the user’s home directory. It can be used, for instance, for 2D plotting of the selected region.

   Links provided in the `CDO <https://easy.gems.dkrz.de/Processing/CDO/index.html>`__ section give a lot of information on possible CDO commands.



.. admonition:: Remark

   For every user there are three directories in which files can be stored. On `DKRZ documentation pages <https://docs.dkrz.de/doc/levante/file-systems.html>`__ you can familiarise yourself with the particularities of each and also with the type of data they are meant for. Pay particular attention to the different data lifetimes!


.. _beginner_dpp0029_jupyter_notebook:

Jupyter notebook
~~~~~~~~~~~~~~~~~~~~~

For creating and running Python, Julia or R scripts for files within Levante JupiterHub is used (see the `MPI Wiki <https://wiki.mpimet.mpg.de/doku.php?id=analysis:postprocessing_icon:0._computing_inf:python:start>`__  for basic access and setup information). Per default upon logging in and choosing a computing time option, a user’s home directory is shown. There, one can create a new script by choosing the notebook type in the upper-right corner. We will be working with the Python 3 unstable option from now on.

Starting out
""""""""""""""

At the beginning of the script we load some basic packages

.. code:: python

   from getpass import getuser # Libaray to copy things
   from pathlib import Path # Object oriented libary to deal with paths
   import os
   from tempfile import NamedTemporaryFile, TemporaryDirectory # Creating temporary Files/Dirs
   from subprocess import run, PIPE
   import sys

   import dask # Distributed data libary
   from dask_jobqueue import SLURMCluster # Setting up distributed memories via slurm
   from distributed import Client, progress, wait # Libaray to orchestrate distributed resources
   import xarray as xr # Libary to work with labeled n-dimensional data and dask

   from dask.utils import format_bytes
   import numpy as np # Pythons standard array library

   import multiprocessing
   from subprocess import run, PIPE
   import warnings
   warnings.filterwarnings(action='ignore')
   import pandas as pd

   dask.config.config.get('distributed').get('dashboard').update({'link':'{JUPYTERHUB_SERVICE_PREFIX}/proxy/{port}/status'})

then create a shortcut for the SCRATCH directory and set up a dask cluster for partitioning the computing

.. code:: python

   # Set some user specific variables
   scratch_dir = Path('/scratch') / getuser()[0] / getuser() # Define the users scratch dir
   # Create a temp directory where the output of distributed cluster will be written to, after this notebook
   # is closed the temp directory will be closed
   dask_tmp_dir = TemporaryDirectory(dir=scratch_dir, prefix='PostProc')
   cluster = SLURMCluster(memory='500GiB',
                          cores=72,
                          project='xz0123',
                          walltime='1:00:00',
                          queue='compute',
                          name='PostProc',
                          scheduler_options={'dashboard_address': ':12435'},
                          local_directory=dask_tmp_dir.name,
                          job_extra=[f'-J PostProc',
                                     f'-D {dask_tmp_dir.name}',
                                     f'--begin=now',
                                     f'--output={dask_tmp_dir.name}/LOG_cluster.%j.o',
                                     f'--output={dask_tmp_dir.name}/LOG_cluster.%j.o'
                                    ],
                          interface='ib0')

..
   (I’M NOT SURE WHAT HAPPENS HERE)

where :code:`xz0123` is replaced by your project’s name, and finally initialise a dask client

.. code:: python

   cluster.scale(jobs=2)
   dask_client = Client(cluster)
   #dask_client.wait_for_workers(18)
   cluster



.. admonition:: Remark

   Clicking on the dashboard link in the output of this cell opens a visualisation of the tasks that are being done when the script is ran. This is helpful to track the progress (for longer tasks), obtain a feeling for the duration of certain operations and locate problems.

Reading
""""""""""""""

.. admonition:: Example

   Code for opening and concatenating dpp0029 output files:

   .. code:: python

      data_path = Path('/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0029/')
      glob_pattern_2d = 'atm3_2d_ml'

      # Collect all file names with pathlib's rglob
      file_names = sorted([str(f) for f in data_path.rglob(f'*{glob_pattern_2d}*.nc')])[1:]
      dset = xr.open_mfdataset(file_names, combine='by_coords', parallel=True)
      dset



.. admonition:: Example

   Code for opening files from the SCRATCH directory:

   .. code:: python

      file_names_s = Path('/scratch/x/x123456') / Path('File.nc')

   where :code:`x123456` stands for your Levante ID.



Bulk of the script
""""""""""""""""""""""""""""

In the bulk of the script one can also process the data, similarly to CDO. The output read from the files is of type :code:`xarray`. Some basic tools for manipulating an object :code:`da` of this type include :code:`da.sel`, :code:`da.where`, :code:`da.mean`, :code:`da.resample`, see examples of use below.

.. admonition:: Example

   Code for selecting variables

   .. code:: python

      var_names = [ 'tas', 'uas']
      dset_subset = dset[var_names]

      #view the new data array
      dset_subset



Changing the time from numerical values to a datetime format facilitates further analysis. It can be done in the following way.

.. admonition:: Example

   Attaching the time to output with certain (fixed) time intervals, here 15 minutes

   .. code:: python

      dset_subset = dset_subset.sel(time=slice(20210121., 20210229.))
      timesf1 = pd.date_range(start='2020-01-21', periods=dset_subset.time.sel(time=slice(20210121., 20210229.)).size ,freq='15min')
      dset_subset['time']=timesf1
      dset_subset

   Attaching the time to output with varying time intervals, here 6 hours and 1 day

   .. code:: python

      timesf1 = pd.date_range(start='2020-01-21', periods=dset_subset.time.sel(time=slice(20200121., 20200301.)).size ,freq='6H')

      #integers can be converted directly to days
      times_old = dset_subset.time.sel(time=slice(20200302., 20210120.)).values
      timesf2_old = pd.to_datetime(times_old,format='%Y%m%d')
      timesf_total = timesf1.union(timesf2_old)
      dset_subset['time'] = timesf_total

      #build daily means wherever the intervals are smaller to homogenise the data
      #the option skipna deletes possible missing values
      dset_subset = dset_subset.resample(time="1D", skipna=True).mean()


Attaching a grid to the cells can be of similar importance for further calculations.

.. admonition:: Example

   Code for attaching heights for a subset **over sea**

   .. code:: python

      file3 = '/work/mh0287/k203123/Dyamond++/icon-aes-dyw2/experiments/dpp0016_atm_vgrid_ml.nc'
      dsetvgrid = xr.open_dataset(file3)
      dsetvgrid

      #selecting a subgrid if necessary
      dsvg_sub = dsetvgrid['zg'].sel(height_2 = slice(14.0, 90.0))

      #overwriting the height levels with geographic heights at a random point over sea
      dset_subset['height']=dsvg_sub[:,1343333].values
      dset_subset

   Note that we have chosen heights in the middle of the corresponding cells. For other possible choices call :code:`dsetvgrid` to access all height variables.

   Code for attaching horizontal coordinates and a land-sea mask

   .. code:: python

      file2 = '/pool/data/ICON/grids/public/mpim/0015/icon_grid_0015_R02B09_G.nc'
      dsetlm = xr.open_dataset(file2)
      dsetlm = dsetlm.rename({'cell' : 'ncells'})
      land_sea_mask = dsetlm.cell_sea_land_mask.persist()
      progress(land_sea_mask ,notebook=False)
      mask = land_sea_mask.values
      dset_subset['land_sea_mask'] = (('ncells'),mask)

      clat = land_sea_mask.clat.values
      clon = land_sea_mask.clon.values
      dset_subset = ds_subset.assign_coords(clon=("ncells",clon),clat=("ncells",clat))
      dset_subset

   Now we can cut out boxes according to latitude and longitude via (for instance)

   .. code:: python

      dset_subset = dset_subset.where((land_sea_mask.clat>=(12.0/180*np.pi)) & (land_sea_mask.clat<=(14.5/180*np.pi)) & (land_sea_mask.clon>=(-59/180*np.pi))& (land_sea_mask.clon<=(-56.5/180*np.pi)), drop = True)

   or select the land part

   .. code:: python

      dset_subset = dset_subset.where(dset_subset.land_sea_mask>0, drop=True)

   Here, as above, :code:`drop=True` is used to delete the missing values.

Here are some more examples of basic operators for :code:`xarray` objects.

.. admonition:: Examples

   Code for calculating the spatial mean:

   .. code:: python

      dset_subset_mean = dset_subset.mean(dim='ncells')

   Omitting a particular date:

   .. code:: python

      dset_subset_clean = dset_subset.drop_sel(time=np.datetime64('2020-03-01'))

   Performing a vectorwise calculation and adding the resulting vector to an existing data array:

   .. code:: python

      #library for numerical computations
      import numpy as np

      #calculate the horizontal wind speed
      dset_subset['ws'] = np.sqrt(dset_subset['va']*dset_subset['va'] + dset_subset['ua']*dset_subset['ua'])

One of the most important tasks performed in the notebook is plotting the preprocessed output. The simplest way is to use the intrinsic plotting method of :code:`xarray` objects, that is, for an object :code:`da` of this type running

:code:`da.plot()`

This method automatically chooses a type of plot suitable for the data structure and produces a simple plot. More sophisticated plots can be created manually. In particular, one can use a cartography library by running

.. code:: python

   from cartopy import crs as ccrs

and produce maps. Some examples can be found in the `MPI Wiki <https://wiki.mpimet.mpg.de/doku.php?id=analysis:postprocessing_icon:3.plottting:python:start>`__. For 2D plots one can import the standard plotting library

.. code:: python

   from matplotlib import pyplot as plt

and use its numerous options.

Saving
""""""""""""""""""""""""""""

You should store preprocessed data that you are working with, especially reduced data sets that need a long time to be calculated.

.. admonition:: Example

   Code for saving data in the SCRATH directory

   .. code:: python

      scratch_dir = Path('/scratch') / getuser()[0] / getuser() #if it has not been defined before
      out_file = Path(scratch_dir) / 'OutfileName.nc'
      dset_subset.to_netcdf(out_file, mode='w')

Example notebook
""""""""""""""""""""""""""""

.. toctree::

   Example Jupyter notebook <Tutorial01.ipynb>
