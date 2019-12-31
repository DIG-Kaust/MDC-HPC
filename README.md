# Reproducible material for EAGE2020 Abstract: "Implementation of large-scale integral operators with modern HPC solutions"
## Authors: M.Ravasi, I.Vasconcelos

All the figures in the abstract can be reproduced by running the following notebooks:

- ``Marchenko3D_datacreation.ipynb``: Data Creation notebook. The synthetic dataset is originally separated into .npz files. In this notebook the dataset is converted into
a singe Zarr file using Dask for concurrent IO. The resulting file will loaded into distributed memory and used for MDC and Marchenko redatuming.

- ``MDC_timing.ipynb``: Timing of MDC operation for single virtual source. Used to create figure 2a.

- ``MDCmulti_timing.ipynb``: Timing of MDC operation for multiple virtual sources. Used to create input data for figure 2b.

- ``MDC_timing_comparison.ipynb``: Plotting different timings of MDC operation. Used to create figure 2.

- ``Marchenko3D.ipynb``: 3D Marchenko redatuming for single virtual point. Create Green's functions to be visualized by

- ``Marchenko3Dmulti.ipynb``: 3D Marchenko redatuming for multiple virtual points. Not used in the paper. ``Marchenko3D_comparison.ipynb``

- ``Marchenko3D_comparison.ipynb``: Plotting different timings of MDC operation. Used to create figure 3.


Auxiliary files:

- ``setup-ssh.sh``: Shell script to setup a SSH Dask cluster (note that you will need a ``hostfile.txt`` file in the same directory with the addresses of the nodes you want to
use when setting up the cluster. Refer to https://docs.dask.org/en/latest/setup/ssh.html for more details.
- ``utils.py``: Small Python functions used in various notebooks


**Note:** All notebooks use the ``$STORE_PATH`` enviroment variable to specify the location of the input dataset. Either create this 
environment variable prior to running a notebook or use ``os.environ["STORE_PATH"] = "/path/to/data/"`` directly within the notebook.
