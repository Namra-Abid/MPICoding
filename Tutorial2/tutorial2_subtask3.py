from mpi4py import MPI
import numpy as np
import heapq  # for efficient k-way merge

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Step 1: Generate full shuffled list only on root
if rank == 0:
    # full_list = list(range(10000)) #This will give error

    full_list = np.array(list(range(10000)), dtype='i')  #  NumPy array
    np.random.shuffle(full_list)
else:
    full_list = None

# Step 2: Determine chunk size and scatter data
chunk_size = 10000 // size
local_chunk = np.empty(chunk_size, dtype='i')  # prepare space

# Scatter chunks of the unsorted list
comm.Scatter(full_list, local_chunk, root=0)

# Step 3: Each process sorts its chunk
local_sorted = sorted(local_chunk.tolist())

# Step 4: Gather sorted chunks back to root
gathered = comm.gather(local_sorted, root=0)

# Step 5: Merge all sorted chunks at root
if rank == 0:
    # Use heapq.merge for efficient k-way merge
    final_sorted_list = list(heapq.merge(*gathered))

    # Check if it's fully sorted
    print(f"Fully sorted? {final_sorted_list == sorted(full_list)}")
    print(f"First 10 elements: {final_sorted_list[:10]}")
    print(f"Last 10 elements: {final_sorted_list[-10:]}")
