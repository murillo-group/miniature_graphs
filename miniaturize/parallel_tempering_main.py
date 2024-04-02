#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

Parallel tempering definition

REQUIRES: Metropolis.py

to run:

mpiexec -n $number_of_nodes python parallel_tempering_main.py
'''

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

    if not os.path.exists('./Results/run_data/'):
        os.makedirs('./Results/run_data/',exist_ok=True)


if size%2:
    raise Exception('An even number of cores is required')

##################################
# Set up variables for each rank #
##################################

#
#beta_0 = np.array([float(sys.argv[2])],dtype=np.float64)
#epsilon = np.array([float(sys.argv[3])],dtype=np.float64)

#my_beta = beta_0 + rank*epsilon

# array of temps for PT, select yours via rank
init_beta = np.array([1.90472625, 1.53273078, 1.16073531, 0.78873985, 0.41674438, 0.04474891])[rank]
my_beta = np.array([init_beta],dtype=np.float64)

STEPS_PER_TEMP_SWAP = 20#int(sys.argv[4])
TOTAL_STEPS = 20000

# load the file with attribute data
with open(sys.argv[1],'r') as json_file:
   targets_and_weights = json.load(json_file)


# each rank creates an initial graph
initial_graph = nx.erdos_renyi_graph(600, .01)
#initial_graph = nx.watts_strogatz_graph(400, 4, .5)

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

# saving data
assorts = np.zeros(TOTAL_STEPS+1)
densitys = np.zeros(TOTAL_STEPS+1)
clusts = np.zeros(TOTAL_STEPS+1)
energys = np.zeros(TOTAL_STEPS+1)
betas = np.zeros(TOTAL_STEPS+1)

assorts[0] = (nx.degree_assortativity_coefficient(replica.get_graph())+1)/2
densitys[0] = nx.density(replica.get_graph())
clusts[0] = nx.average_clustering(replica.get_graph())
energys[0] = replica.get_energy(**targets_and_weights)
betas[0] = my_beta[0]

# loop over a number of iterations
for step in range(TOTAL_STEPS):

    # each replica runs a single step
    replica.run_step(targets_and_weights)

    # each replica updates its energy
    my_energy[0] = replica.get_energy(**targets_and_weights)


    assorts[step+1] = (nx.degree_assortativity_coefficient(replica.get_graph())+1)/2
    densitys[step+1] = nx.density(replica.get_graph())
    clusts[step+1] = nx.average_clustering(replica.get_graph())
    energys[step+1] = replica.get_energy(**targets_and_weights)
    betas[step+1] = my_beta[0]

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
    filename = "./Results/graphs/"+sys.argv[1]+'_'+str(my_beta)+"_graph{}.npz"
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    # save the adjacency matrix corresponding to the minimum energy
    save_npz(filename, nx.adjacency_matrix(replica.get_graph()))



counter = 0
filename = "./Results/run_data/assorts_"+str(init_beta)+"_{}.npz"
while os.path.isfile(filename.format(counter)):
    counter += 1

np.save('Results/run_data/assorts_'+str(init_beta)+'_'+str(counter),assorts)
np.save('Results/run_data/densitys_'+str(init_beta)+'_'+str(counter),densitys)
np.save('Results/run_data/clusts_'+str(init_beta)+'_'+str(counter),clusts)
np.save('Results/run_data/energys_'+str(init_beta)+'_'+str(counter),energys)
np.save('Results/run_data/betas_'+str(init_beta)+"_"+str(counter),betas)
