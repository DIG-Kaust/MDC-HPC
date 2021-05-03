# Reproducible material for "Implementation of large-scale integral operators with modern HPC solutions"
## Authors: M.Ravasi, I.Vasconcelos

This repository contains the material used in the *Implementation of large-scale integral operators with modern HPC solutions* extended abstract to be 
presented at EAGE 2020 in Amsterdam.

Users interested to reproduce figures in the abstract can simply run the provided notebooks as explained below. Python scripts are also provided for long-running
jobs that perform a certain processing step for the entire dataset.

These codes have been tested on local HPC as well as K8S with similar performances


### Notebooks

All the figures in the abstract can be reproduced by running the following notebooks:

- ``Marchenko3D_datacreation.ipynb``: Data Creation notebook. The synthetic dataset is originally separated into .npz files. In this notebook the dataset is converted into
a singe Zarr file using Dask for concurrent IO. The resulting file will loaded into distributed memory and used for MDC and Marchenko redatuming.

- ``Marchenko3D_datavisualization_time_frequency.ipynb``: Data Visualization notebook. Data in time-space, fkk, and fxy (frequencies slice) is displayed.

- ``MDC_timing.ipynb``: Timing of MDC operation for single virtual source. Used to create figure 2a.

- ``MDCmulti_timing.ipynb``: Timing of MDC operation for multiple virtual sources. Used to create input data for figure 2b.

- ``MDC_timing_comparison.ipynb``: Plotting different timings of MDC operation. Used to create figure 2.

- ``Marchenko3D.ipynb``: 3D Marchenko redatuming for single virtual point. Create Green's functions to be visualized by

- ``Marchenko3Dmulti.ipynb``: 3D Marchenko redatuming for multiple virtual points. Not used in the paper.

- ``Marchenko3D_comparison.ipynb``: Plotting Marchenko fields for different subsampling factors. Used to create figure 3.

- ``MDD3D.ipynb``: 3D Multi-dimensional deconvolution of Marchenko fields. Not used in the abstract.

- ``MDD3D_visualization.ipynb``: Plotting MDD redatumed local responses. Not used in the abstract.


### Python scripts

- ``MDC_timing.py``: Timing of MDC operation for single virtual source. Same as ``MDC_timing.ipynb``, used to perform timing of several configurations in 'batch' mode together with ``MDC_timing.sh``

- ``MDCmulti_timing.py``: Timing of MDC operation for multiple virtual sources. Same as ``MDCmulti_timing.ipynb``, used to perform timing of several configurations in 'batch' mode together with ``MDCmulti_timing.sh``

- ``Marchenko3D.py``: 3D Marchenko redatuming for an entire depth level. Used to estimate fields in 'batch' mode together with ``Marchenko3D.sh``

- ``Marchenko3Dmulti.py``: 3D Marchenko redatuming for an entire depth level using multiple virtual points. Used to estimate fields in 'batch' mode together with ``Marchenko3Dmulti.sh``

- ``MDD3D.py``: 3D Multi-dimensional deconvolution of Marchenko fields for an entire depth level. Used to estimate fields in 'batch' mode together with ``MDD3D.sh``


### Auxiliary files:

- ``setup-ssh.sh``: Shell script to setup a SSH Dask cluster (note that you will need a ``hostfile.txt`` file in the same directory with the addresses of the nodes you want to
use when setting up the cluster. Refer to https://docs.dask.org/en/latest/setup/ssh.html for more details.
- ``utils.py``: Small Python functions used in various notebooks

### K8S

This directory contains the helm charts used to configure K8S and some instructions on how to run the ``Marchenko3D.py``
in a Kubernetes cluster. Note that our setup is based on the https://github.com/dask/helm-chart and we refer to those
for more details with respect to the entire configuration setup.

### Input data:

The input dataset has been created using a finite-difference modelling code available in the
[Madagascar](http://www.ahay.org/wiki/Main_Page) sofware package. Given the size of the input dataset,
authors will not be able to share it directly but can provide the ``SConstruct`` file used to create the data.

### Environment

To ensure reproducibility of the results, we suggest using the ``requirements.txt`` file when creating an environment.

**Note:** All notebooks use the ``$STORE_PATH`` enviroment variable to specify the location of the input dataset.
Either create this environment variable prior to running a notebook or use ``os.environ["STORE_PATH"] = "/path/to/data/"`` directly within the notebook.
