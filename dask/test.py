import numpy as np
import dask.array as da

# a = np.ones([10000, 1000, 1000])
# print(a)                                # Does not load (too big)

# a = da.ones(10000, 1000, 1000)
# print(a)                                # works now



# Clients and docker
# from dask.distributed import Client
# from dask_kubernetes import KubeCluster
# cluster = KubeCluster(
#     pod_template={"spec": {"containers": [{"image": "your-dask-worker-image", "name": "dask-worker"}]}})

# client = Client(cluster)







































