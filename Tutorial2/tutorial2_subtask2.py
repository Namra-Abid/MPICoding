from mpi4py import MPI
import numpy as np
from datetime import datetime

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# For performance testing
N_REPEAT = 100_000

# Root initializes data
if rank == 0:
    a = np.array([0, 1, 2, 3, 4, 5], dtype='i')
    b = np.array([6, 7, 8, 9, 10, 11], dtype='i')
else:
    a = None
    b = None

# Determine chunk size
vec_len = 6  # Must match length of a and b
chunk_size = vec_len // size

# Create local arrays for each process
local_a = np.empty(chunk_size, dtype='i')
local_b = np.empty(chunk_size, dtype='i')

# Scatter chunks to all processes
comm.Scatter([a, chunk_size, MPI.INT], local_a, root=0)
comm.Scatter([b, chunk_size, MPI.INT], local_b, root=0)

# Time measurement (only once for whole 100k loops)
if rank == 0:
    start = datetime.now()

local_dot = 0
for _ in range(N_REPEAT):
    local_dot += np.dot(local_a, local_b)

# Reduce all partial results to rank 0
total_dot = comm.reduce(local_dot, op=MPI.SUM, root=0)

# Print final result and time (only on root)
if rank == 0:
    print(f"Final dot product (after {N_REPEAT} repeats): {total_dot}")
    print(f"Took {datetime.now() - start}")
