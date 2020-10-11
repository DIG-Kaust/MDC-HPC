# Running codes on Azure K8S
This directory contains sample setups for running our Dask-powered codes on K8S on
Microsoft Azure cloud (although most of the configuration should work in other
clouds too).

## Objective

In the ``charts`` directory, two helm charts are created to satisfy different
working scenarios:

- Interactive development: the chart is taken from https://github.com/dask/helm-chart
and some of the templates files are modified to allow persistent volumes (PV) to be
included in the deployment of the dask-k8s solution. The PV is shared among the workers
pods (as well as jupyter pod) and is used to store input data as well as persist
output data.

- Batch jobs: the previous chart is modified to allow running jobs in batch.
A new docker image is created to be used on a `master` pod that has the task
to runs a chosen script in batch mode (this replaces the `jupyter` pod).
Different scripts available in the `scripts` directory are also copied to
the docker image. These charts are configured to run on Spot instances where
restart may happen if a node is evicted.

In the ``deployment`` directoy, two sample values files are also provided.

## Cluster

To use `Standard_E16_v3` istances, a nodepool can be created running the following command:

```
az aks nodepool add --name e16regular \
    --cluster-name ${CLUSTER_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --max-pods 100 \
    --node-count 4 \
    --node-osdisk-size 256 \
    --node-vm-size Standard_E16_v3
```

To use `Standard_E16_v3` Spot istances, the command is modified as explained
in https://github.com/gkaleta/AKS-Spot-Instances:

```
az aks nodepool add --name e16spot \
    --cluster-name ${CLUSTER_NAME} \
    --resource-group ${RESOURCE_GROUP} \
    --node-count 4 \
    --node-osdisk-size 256 \
    --node-vm-size Standard_E16_v3 \
    --priority Spot \
    --spot-max-price 0.2388 \
    --verbose
```

## Setup
 
To install our helm charts simply run:

```
helm upgrade --install --namespace ${NAMESPACE} dask ./charts/dask --values ./deployment/dask-values.yaml
```

or

```
helm upgrade --install --namespace ${NAMESPACE} dask_batch ./charts/dask_batch --values ./deployment/dask-batch-values.yaml
```

When changes are made to the `dask-values.yaml` (or to any template), simply re-run the above command to upgrade the chart.


To check on the status of the chart simply run `helm list`. Similarly to see if the pods have been already deployed run `kubectl -n ${NAMESPACE} get pods`


## Sample code

The script `Marchenko.py` is configured to run as a batch job. It is mostly meant to show
how we can run a Dask-powered workflow keeping track of its progress to be relisient
to node eviction (especially important when using Spot istances).