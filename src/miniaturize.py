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

DATA_DIR = os.environ['DATA_DIR']
directory_name = sys.argv[1]
graph_name = sys.argv[2]

# rank zero makes directory if it hasn't been made yet
if rank == 0:
    if not os.path.exists('../data/'+directory_name):
        os.makedirs('../data/'+directory_name,exist_ok=True)

    if not os.path.exists('../results/'+directory_name+'/mini_data/'):
        os.makedirs('../results/'+directory_name+'/mini_data/',exist_ok=True)


if size%2:
    raise Exception('An even number of cores is required')

##################################
# Set up variables for each rank #
##################################

# Initialize buffers
beta_arr = np.array([0.5,0.75,1,1.25,1.5,2.0]) * beta
my_beta = np.array([beta_arr[rank]])
neighbors_beta = np.empty(1,dtype=np.float64)

my_energy = np.empty(1,dtype=np.float64)
neighbors_energy = np.empty(1,dtype=np.float64)


n_substeps = 20
n_steps = 100

# Load target metrics
file_metrics = os.path.join(DATA_DIR,'networks',graph_name,'metrics.json')
with open(file_metrics) as json_file:
   metrics = json.load(json_file)

# Initialize graph
G = nx.erdos_renyi_graph(600,metrics['density'])

#######################################
# Give each rank a metropolis replica #
#######################################
n_cycles = (n_steps // n_substeps) + 1
n_steps_last = n_steps % n_substeps

replica = Metropolis(G, 
                     my_beta,
                     funcs_metrics,
                     n_substeps,
                     metrics_weights=weights,
                     n_changes=10
                     )


def exchange() -> float:
    '''Replicas exchange temperatures according to their energies
    '''
    def prob(E0,E1,B0,B1) -> float:
        '''Defines exchange probability according to the metropolis criterion
        '''
        return min(1.0,np.exp((E0-E1)*(B0-B1)))
    
    def swap(flag_send,flag_recv,ref):
        '''Even ranks propose energy change in the specified direction
        '''
        if flag_send[rank]:
            # Send energy and temp forward to next rank
            comm.Send([my_energy, MPI.DOUBLE], dest=rank+ref, tag=0)
            comm.Send([my_beta, MPI.DOUBLE], dest=rank+ref, tag=1)
            
            # Receive new temp
            comm.Recv([my_beta, MPI.DOUBLE], source=rank+ref, tag=2)

        elif flag_recv[rank]:
            # Receive energy and temp from previous rank
            comm.Recv([neighbors_energy, MPI.DOUBLE], source=rank-ref, tag=0)
            comm.Recv([neighbors_beta, MPI.DOUBLE], source=rank-ref,tag=1)
            
            # Accept or reject switch
            if prob(my_energy,neighbors_energy,my_beta,neighbors_beta) > np.random.uniform(0,1):
                comm.Send([my_beta, MPI.DOUBLE], dest=rank-ref, tag=2)
                my_beta = neighbors_beta.copy()
                
            else: 
                comm.Send([neighbors_beta, MPI.DOUBLE], dest=rank-ref, tag=2)
        
        # Sync replicas
        comm.Barrier()
        
    # Get indices of senders
    flag_send = (np.arange(0,len(beta_arr)) % 2).astype(bool)
    flag_recv = np.invert(flag_send)
    
    # Send forward
    swap(flag_send, flag_recv, ref=1)
    
    # Send backward
    flag_send[0] = False
    flag_recv[-1] = False
    swap(flag_send, flag_recv, ref=-1)
    
for cycle in range(n_cycles):
    # Update number of iterations for the last cycle
    if cycle == (n_cycles-1):
        replica.n_iterations = n_steps_last
        
    # Transform graph
    replica.transform(G,metrics_target)
    
    # Retrieve trajectories
    trajectories = replica.trajectories__
    
    # Swap temperatures
    my_energy = trajectories['Energy'][-1]
    my_beta = replica.beta
    exchange()
    
    replica.beta = my_beta
    
    
    
    



# # loop over a number of iterations
# for step in range(TOTAL_STEPS):

#     # each replica runs a single step
#     replica.run_step(targets_and_weights)

#     # each replica updates its energy
#     my_energy[0] = replica.get_energy(**targets_and_weights)

#     assorts[step+1] = (nx.degree_assortativity_coefficient(replica.get_graph())+1)/2
#     densitys[step+1] = nx.density(replica.get_graph())
#     clusts[step+1] = nx.average_clustering(replica.get_graph())
#     energys[step+1] = replica.get_energy(**targets_and_weights)
#     betas[step+1] = my_beta[0]

#     # only attempt a temp swap after X steps
#     if not step % STEPS_PER_TEMP_SWAP:

#        ###########################
#        # Even ranks send forward #
#        ###########################

#         # even ranks2
#         if not rank%2:
#             #print(f'rank {rank} sending to rank {rank+1}')
#             # send energy and temp forward a rank
#             comm.Send([my_energy, MPI.DOUBLE], dest=rank+1, tag=0)
#             comm.Send([my_beta, MPI.DOUBLE], dest=rank+1, tag=1)

#             # get beta for next step
#             comm.Recv([my_beta, MPI.DOUBLE], source=rank+1, tag=2)

#         else:
#             # receive energy and temp from back a rank
#             comm.Recv([neighbors_energy[0:1], MPI.DOUBLE], source=rank-1, tag=0)
#             comm.Recv([neighbors_beta[0:1], MPI.DOUBLE], source=rank-1, tag=1)

#             # probability of a switch
#             p = np.min([ [1.0], np.exp( (my_energy - neighbors_energy)*(my_beta - neighbors_beta) ) ])

#             # if switching send your beta
#             if p > np.random.uniform(0,1):
#                 comm.Send([my_beta, MPI.DOUBLE], dest=rank-1, tag=2)
#                 my_beta = neighbors_beta.copy()
#             # otherwise send their beta back
#             else:
#                 comm.Send([neighbors_beta, MPI.DOUBLE], dest=rank-1, tag=2)
#         # Wait to ensure all switching has occurred (need to check if this is needed)
#         comm.Barrier()
#         ############################
#         # Even ranks send backward #
#         ############################

#         if rank > 0 and not rank%2:
#             #print(f'rank {rank} sending to rank {rank-1}')
#             comm.Send([my_energy, MPI.DOUBLE], dest=rank-1, tag=0)
#             comm.Send([my_beta, MPI.DOUBLE], dest=rank-1, tag=1)

#             comm.Recv([my_beta, MPI.DOUBLE], source=rank-1, tag=2)


#         if rank<size-1 and rank%2:
#             comm.Recv([neighbors_energy, MPI.DOUBLE], source=rank+1, tag=0)
#             comm.Recv([neighbors_beta, MPI.DOUBLE], source=rank+1, tag=1)

#             p =  np.min([ [1.0], np.exp( (my_energy - neighbors_energy)*(my_beta - neighbors_beta) ) ])

#             # if switching send your beta
#             if p > np.random.uniform(0,1):
#                 comm.Send([my_beta, MPI.DOUBLE], dest=rank+1, tag=2)
#                 my_beta = neighbors_beta.copy()
#             # otherwise send their beta back
#             else:
#                 comm.Send([neighbors_beta, MPI.DOUBLE], dest=rank+1, tag=2)

# # store your energy and rank
# my_energy_core = np.array([(my_energy,rank)],dtype=[('energy',np.float64), ('rank',np.int32)])
# # allocate memory for min energy and rank
# min_energy_core = np.empty(1,dtype=[('energy',np.float64), ('rank',np.int32)])

# # reduce to all ranks the mimum energy and rank that has the mimimum
# comm.Allreduce([my_energy_core, 1, MPI.DOUBLE_INT], [min_energy_core, 1, MPI.DOUBLE_INT], op=MPI.MINLOC)


# # the rank with the minimum saves their adjacency matrix
# if rank == min_energy_core['rank']:

#     # create a file that doesn't exist yet
#     counter = 0
#     filename = "../data/"+directory_name+"/"+graph_name+'_mini'
#     # save the adjacency matrix corresponding to the minimum energy
#     save_npz(filename, nx.adjacency_matrix(replica.get_graph()))

# np.save('../results/'+directory_name+'/mini_data/assorts',assorts)
# np.save('../results/'+directory_name+'/mini_data/densitys',densitys)
# np.save('../results/'+directory_name+'/mini_data/clusts',clusts)
# np.save('../results/'+directory_name+'/mini_data/energys',energys)
# np.save('../results/'+directory_name+'/mini_data/betas',betas)
