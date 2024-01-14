
from dask.distributed import Client
import dask.array as da
import numpy as np
import webbrowser

def complex_computation(x):
    # Perform some complex operations
    y = da.sin(x) ** 2
    z = da.cos(x) ** 2
    return da.sqrt(y + z)

def main():
    client = Client()
    print("Dashboard link:", client.dashboard_link)
    webbrowser.open(client.dashboard_link)

    # Create a large random Dask array
    x = da.random.random(size=(20000, 20000), chunks=(1000, 1000))

    # Perform a complex computation
    result = complex_computation(x)

    # Compute the result
    computed_result = result.compute()
    print(computed_result)



if __name__ == '__main__':
    main()




# Dashboard link: http://127.0.0.1:8787/status
# Dashboard link: http://127.0.0.1:8787/status







# client.close()
# import numpy as np
# import dask.array as da

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





