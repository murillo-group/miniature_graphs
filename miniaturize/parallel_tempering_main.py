from mpi4py import MPI
import numpy as np
from Metropolis import Metropolis
import networkx as nx
from scipy.sparse import save_npz
import os
import sys
import json

#################
# Setup for MPI #
#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

# rank zero makes directory if it hasn't been made yet
if rank == 0:
    if not os.path.exists('./Results/graphs/'):
        os.makedirs('./Results/graphs/',exist_ok=True)


if size%2:
    raise Exception('An even number of cores is required')

##################################
# Set up variables for each rank #
##################################

#
beta_0 = np.array([1000],dtype=np.float64)
epsilon = np.array([1000],dtype=np.float64)

my_beta = beta_0 + rank*epsilon

TOTAL_STEPS = 21
STEPS_PER_TEMP_SWAP = 20

# load the file with attribute data
with open(sys.argv[1],'r') as json_file:
   targets_and_weights = json.load(json_file)


# each rank creates an initial graph
initial_graph = nx.erdos_renyi_graph(400, 117.625/4038)

########################################
# set up buffers for each ranks energy #
########################################

my_energy = np.empty(1,dtype=np.float64)

# create a buffer for n-1 and n+1 neighbors info
neighbors_energy = np.empty(1,dtype=np.float64)
neighbors_beta = np.empty(1,dtype=np.float64)


#######################################
# Give each rank a metropolis replica #
#######################################

replica = Metropolis(initial_graph, my_beta)

# loop over a number of iterations
for step in range(TOTAL_STEPS):

    # each replica runs a single step
    replica.run_step(targets_and_weights)

    # each replica updates its energy
    my_energy[0] = replica.get_energy(**targets_and_weights)

    # only attempt a temp swap after X steps
    if not step % STEPS_PER_TEMP_SWAP:

       ###########################
       # Even ranks send forward #
       ###########################

        # even ranks2
        if not rank%2:
            #print(f'rank {rank} sending to rank {rank+1}')
            # send energy and temp forward a rank
            comm.Send([my_energy, MPI.DOUBLE], dest=rank+1, tag=0)
            comm.Send([my_beta, MPI.DOUBLE], dest=rank+1, tag=1)

            # get beta for next step
            comm.Recv([my_beta, MPI.DOUBLE], source=rank+1, tag=2)

        else:
            # receive energy and temp from back a rank
            comm.Recv([neighbors_energy[0:1], MPI.DOUBLE], source=rank-1, tag=0)
            comm.Recv([neighbors_beta[0:1], MPI.DOUBLE], source=rank-1, tag=1)

            # probability of a switch
            p = np.min([ [1.0], np.exp( (my_energy - neighbors_energy)*(my_beta - neighbors_beta) ) ])

            # if switching send your beta
            if p > np.random.uniform(0,1):
                comm.Send([my_beta, MPI.DOUBLE], dest=rank-1, tag=2)
                my_beta = neighbors_beta.copy()
            # otherwise send their beta back
            else:
                comm.Send([neighbors_beta, MPI.DOUBLE], dest=rank-1, tag=2)
        # Wait to ensure all switching has occurred (need to check if this is needed)
        comm.Barrier()
        ############################
        # Even ranks send backward #
        ############################

        if rank > 0 and not rank%2:
            #print(f'rank {rank} sending to rank {rank-1}')
            comm.Send([my_energy, MPI.DOUBLE], dest=rank-1, tag=0)
            comm.Send([my_beta, MPI.DOUBLE], dest=rank-1, tag=1)

            comm.Recv([my_beta, MPI.DOUBLE], source=rank-1, tag=2)


        if rank<size-1 and rank%2:
            comm.Recv([neighbors_energy, MPI.DOUBLE], source=rank+1, tag=0)
            comm.Recv([neighbors_beta, MPI.DOUBLE], source=rank+1, tag=1)

            p =  np.min([ [1.0], np.exp( (my_energy - neighbors_energy)*(my_beta - neighbors_beta) ) ])

            # if switching send your beta
            if p > np.random.uniform(0,1):
                comm.Send([my_beta, MPI.DOUBLE], dest=rank+1, tag=2)
                my_beta = neighbors_beta.copy()
            # otherwise send their beta back
            else:
                comm.Send([neighbors_beta, MPI.DOUBLE], dest=rank+1, tag=2)

# store your energy and rank
my_energy_core = np.array([(my_energy,rank)],dtype=[('energy',np.float64), ('rank',np.int32)])
# allocate memory for min energy and rank
min_energy_core = np.empty(1,dtype=[('energy',np.float64), ('rank',np.int32)])

# reduce to all ranks the mimum energy and rank that has the mimimum
comm.Allreduce([my_energy_core, 1, MPI.DOUBLE_INT], [min_energy_core, 1, MPI.DOUBLE_INT], op=MPI.MINLOC)


# the rank with the minimum saves their adjacency matrix
if rank == min_energy_core['rank']:

    # create a file that doesn't exist yet
    counter = 0
    filename = "./Results/graphs/"+sys.argv[1]+"_graph{}.npz"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    # save the adjacency matrix corresponding to the minimum energy
    save_npz(filename, nx.adjacency_matrix(replica.get_graph()))

    
