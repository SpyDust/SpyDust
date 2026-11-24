"""
This module adapts the MPI utilities from caput, distributed under GPL version 2.
See LICENSE.caput and https://github.com/radiocosmology/caput.
"""
import numpy as np
import logging

rank = 0
size = 1
_comm = None
world = None
rank0 = True


logger = logging.getLogger(__name__)

from mpi4py import MPI
_comm = MPI.COMM_WORLD
world = _comm

rank = _comm.Get_rank()
size = _comm.Get_size()

if _comm is not None and size > 1:
    logger.debug("Starting MPI rank=%i [size=%i]", rank, size)

rank0 = rank == 0

def partition_list(full_list, i, n, method="con"):
    """
    Partition a list into `n` pieces. Return the `i`th partition.
    """

    def _partition(N, n, i):
        # If partiion `N` numbers into `n` pieces,
        # return the start and stop of the `i` th piece
        base = N // n
        rem = N % n
        num_lst = rem * [base + 1] + (n - rem) * [base]
        cum_num_lst = np.cumsum([0] + num_lst)

        return cum_num_lst[i], cum_num_lst[i + 1]

    N = len(full_list)
    start, stop = _partition(N, n, i)

    if method == "con":
        return full_list[start:stop]
    elif method == "alt":
        return full_list[i::n]
    elif method == "rand":
        choices = np.random.permutation(N)[start:stop]
        return [full_list[i] for i in choices]
    else:
        raise ValueError("Unknown partition method %s" % method)

def partition_list_mpi(full_list, method="con", comm=_comm):
    """Partition collectively, using a single permutation for random assignment."""
    if comm is None:
        return partition_list(full_list, 0, 1, method=method)
    if method == "rand":
        order = np.random.permutation(len(full_list)) if comm.rank == 0 else None
        order = comm.bcast(order, root=0)
        indices = partition_list(order, comm.rank, comm.size)
        return [full_list[i] for i in indices]
    return partition_list(full_list, comm.rank, comm.size, method=method)

def _check_rank_errors(comm, error, operation, item=None):
    location = f"rank {comm.rank}" if item is None else f"rank {comm.rank}, item {item}"
    message = None if error is None else f"{location}: {type(error).__name__}: {error}"
    for failure in comm.allgather(message):
        if failure is not None:
            raise RuntimeError(f"MPI {operation} failed on {failure}") from error


def parallel_map(func, glist, root=None, method="con", comm=_comm):
    """
    Apply a parallel map using MPI.
    Should be called collectively on the same list. All ranks return the full
    set of results.
    Parameters
    ----------
    func : function
        Function to apply.
    glist : list
        List of map over. Must be globally defined.
    root : None or Integer
        Which process should gather the results, all processes will gather the results if None.
    method: str
        How to split `glist` to each process, can be 'con': continuously, 'alt': alternatively, 'rand': randomly. Default is 'con'.
    comm : MPI communicator
        MPI communicator that array is distributed over. Default is the gobal _comm.
    Returns
    -------
    results : list
        Global list of results.
    """

    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        return [func(item) for item in glist]

    # Pair up each list item with its position.
    zlist = list(enumerate(glist))

    # Partition list based on MPI rank
    llist = partition_list_mpi(zlist, method=method, comm=comm)

    # Every rank must reach the same collective even if a local item fails.
    flist = []
    local_error = None
    failed_index = None
    for ind, item in llist:
        try:
            flist.append((ind, func(item)))
        except Exception as exc:
            local_error = exc
            failed_index = ind
            break
    _check_rank_errors(comm, local_error, "parallel_map", failed_index)

    rlist = None
    if root is None:
        # Gather all results onto all ranks
        rlist = comm.allgather(flist)
    else:
        # Gather all results onto the specified rank
        rlist = comm.gather(flist, root=root)

    if rlist is not None:
        # Flatten the list of results
        flatlist = [item for sublist in rlist for item in sublist]

        # Sort into original order
        sortlist = sorted(flatlist, key=(lambda item: item[0]))

        # Synchronize
        # barrier(comm=comm)

        # Extract the return values into a list
        return [item for ind, item in sortlist]
    else:
        return None



def parallel_map_joblib(func, glist, n_jobs=-1, backend='loky'):
    """
    Apply a parallel map using joblib with loky backend.
    
    Parameters
    ----------
    func : function
        Function to apply.
    glist : list
        List to map over.
    n_jobs : int, optional
        Number of parallel jobs. -1 means use all available cores. Default is -1.
    backend : str, optional
        Backend to use for parallelization. Options are 'threading', 'loky', 'multiprocessing'.
        Default is 'loky'.
    
    Returns
    -------
    results : list
        List of results in the same order as input.
    """
    from joblib import Parallel, delayed

    results = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(func)(item) for item in glist
    )
    return results



def parallel_jobs_no_gather_no_return(func, glist, method="con", comm=_comm):
    """Execute distributed jobs for side effects without gathering their results.

    In a single process this legacy interface returns the local results.
    On multiple ranks each job receives zip(*item) and returns are discarded.
    """
    # Synchronize
    barrier(comm=comm)

    # If we're only on a single node, then just perform without MPI
    if comm is None or comm.size == 1:
        return [func(item) for item in glist]


    # Partition list based on MPI rank
    llist = partition_list_mpi(glist, method=method, comm=comm)

    # Operate on sublist
    local_error = None
    try:
        for zipped_item in llist:
            func(zip(*zipped_item))
    except Exception as exc:
        local_error = exc
    _check_rank_errors(comm, local_error, "parallel_jobs_no_gather_no_return")
    return None

def barrier(comm=_comm):
    """
    Synchronize all MPI processes.
    """
    if comm is not None and comm.size > 1:
        comm.Barrier()


#def save_to_hdf5_parallel(file_path, local_data, local_key):

    # Open the HDF5 file in parallel
#    with h5py.File(file_path, 'a', driver='mpio', comm=_comm) as file:
        # Create a dataset with collective mode (MPI)
#        dataset = file.create_dataset(local_key, data=local_data, chunks=True)

    # Ensure all processes have finished writing before proceeding
#    barrier()
