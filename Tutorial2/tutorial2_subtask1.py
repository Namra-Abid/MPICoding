from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # Unique ID of this process
size = comm.Get_size() # # Total number of processes in this run

# Input vectors on root process
if rank == 0:
    a = np.array([0, 1, 2, 3, 4, 5], dtype='i')
    b = np.array([6, 7, 8, 9, 10, 11], dtype='i')

    # Split vectors among workers
    chunks_a = np.array_split(a, size)
    chunks_b = np.array_split(b, size)

    # Send chunks to workers
    for i in range(1, size):
        comm.send(chunks_a[i], dest=i)
        comm.send(chunks_b[i], dest=i)

    # Root also computes its part
    local_dot = np.dot(chunks_a[0], chunks_b[0])
    
    # Receive partial results
    for i in range(1, size):
        partial_dot = comm.recv(source=i)
        local_dot += partial_dot

    print(f"Final dot product: {local_dot}")

else:
    # Receive chunks
    a_part = comm.recv(source=0)
    b_part = comm.recv(source=0)

    # Compute local dot product
    local_dot = np.dot(a_part, b_part)

    # Send result back to root
    comm.send(local_dot, dest=0)
