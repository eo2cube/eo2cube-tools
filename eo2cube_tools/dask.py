from importlib.util import find_spec
import os
import dask
from IPython.display import display
from datacube.utils.dask import start_local_dask

_HAVE_PROXY = bool(find_spec("jupyter_server_proxy"))


def create_local_dask_cluster(
    workers=1, threads=None, mem_limit=None, spare_mem="20Gb", display_client=True
):
    """
    Using the datacube utils function `start_local_dask`, generate
    a local dask cluster.

    Parameters
    ----------
    spare_mem : String, optional
        The amount of memory, in Gb, to leave for the notebook to run.
        This memory will not be used by the cluster. e.g '3Gb'
    display_client : Bool, optional
        An optional boolean indicating whether to display a summary of
        the dask client, including a link to monitor progress of the
        analysis. Set to False to hide this display.
    workers: int
        Number of worker processes to launch
    threads: int, optional
        Number of threads per worker, default is as many as there are CPUs
    mem_limit: String, optional
        Maximum memory to use across all workers
    """

    if _HAVE_PROXY:
        # Configure dashboard link to go over proxy
        prefix = os.environ.get("JUPYTERHUB_SERVICE_PREFIX", "/")
        dask.config.set({"distributed.dashboard.link": prefix + "proxy/{port}/status"})

    # Start up a local cluster
    client = start_local_dask(
        n_workers=workers,
        threads_per_worker=threads,
        memory_limit=mem_limit,
        mem_safety_margin=spare_mem,
    )

    # Show the dask cluster settings
    if display_client:
        display(client)
