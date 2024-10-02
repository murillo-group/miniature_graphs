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
from minigraphs import Metropolis
import networkx as nx
from scipy.sparse import save_npz
import os
import sys
import json
import pandas as pd
import datetime

#################
# Setup for MPI #
#################

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.size

DATA_DIR = os.environ['DATA_DIR']
graph_name = sys.argv[1]
n_vertices = int(sys.argv[2])
density = float(sys.argv[3])

if size % 2:
    raise Exception('An even number of cores is required')

##################################
# Set up variables for each rank #
##################################

# Initialize buffers
beta_arr = np.array([0.5,0.75,1,1.25,1.5,1.75])

n_substeps = 50
n_steps = 10000

# Load target metrics 
file_name_metrics = os.path.join(DATA_DIR,'networks',graph_name,'metrics.json')
with open(file_name_metrics,'r') as metrics_file:
   metrics_target = json.load(metrics_file)
   
# Load parameters
file_name_params = os.path.join(DATA_DIR,'params',graph_name,f'params_{n_vertices}.json')
with open(file_name_params,'r') as params_file:
    params = json.load(params_file)

# Initialize annealer variables
beta_opt = params['beta']
B0 = beta_arr[rank] * beta_opt
metrics_funcs = {
    'density': nx.density,
    'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
    'clustering': nx.average_clustering
}
metrics_target = {key:metrics_target[key] for key in metrics_funcs.keys()}
weights = {key:params['weights'][key] for key in metrics_funcs.keys()}

if rank == 0:
    print(f"Miniaturizing '{graph_name}' to size {n_vertices}...")
    print(f"Beta opt: {beta_opt}\n")
    print(f"\t - Target metrics: {metrics_target}")
    print(f"\t - Weights: {weights}")
    print(f"\t - Functions: {metrics_funcs}")
    

# Initialize graph at the specified density
G = nx.erdos_renyi_graph(n_vertices,density)

#######################################
# Give each rank a metropolis replica #
#######################################
n_cycles = int(np.ceil(n_steps / n_substeps))
cycles = [n_substeps] * (n_steps // n_substeps)

remainder = n_steps % n_substeps
if remainder != 0:
    cycles += [remainder]

replica = Metropolis(B0,
                     metrics_funcs,
                     n_substeps,
                     metrics_weights=weights,
                     n_changes=10
                     )

def exchange(E0: float,
             B0: float
            ) -> float:
    '''Replicas exchange temperatures according to their energies
    '''
    def prob(E0,E1,B0,B1) -> float:
        '''Defines exchange probability according to the metropolis criterion
        '''
        return min(1.0,np.exp((E0-E1)*(B0-B1)))
    
    def swap(E0,B0,flag_send,flag_recv,ref) -> float:
        '''Even ranks propose energy change in the specified direction
        '''
        if flag_send[rank]:
            # Send energy and temp forward to next rank
            comm.send(E0, dest=rank+ref, tag=0)
            comm.send(B0, dest=rank+ref, tag=1)
            
            # Receive new temp
            B0 = comm.recv(source=rank+ref, tag=2)

        elif flag_recv[rank]:
            # Receive energy and temp from previous rank
            E1 = comm.recv(source=rank-ref, tag=0)
            B1 = comm.recv(source=rank-ref, tag=1)
            
            # Accept or reject switch
            if prob(E0,E1,B0,B1) > np.random.uniform(0,1):
                comm.send(B0, dest=rank-ref, tag=2)
                B0 = B1
                
            else: 
                comm.send(B1, dest=rank-ref, tag=2)
                 
        # Sync replicas
        comm.Barrier()
        
        return B0
        
    # Get indices of senders
    flag_recv = (np.arange(0,len(beta_arr)) % 2).astype(bool)
    flag_send = np.invert(flag_recv)
    
    # Send forward
    B0 = swap(E0,B0,flag_send, flag_recv, ref=1)
    
    # Send backward
    flag_send[0] = False
    flag_recv[-1] = False
    B0 = swap(E0,B0,flag_send, flag_recv, ref=-1)
    
    return B0
    
for cycle,steps in enumerate(cycles):
    if rank == 0:
        print(f"\nCycle {cycle+1}/{len(cycles)} <<<\n")
        verbose = True 
    else:
        verbose = False
        
    # Update number of iterations for the last cycle
    replica.n_iterations = steps
        
    # Transform graph
    replica.transform(G,metrics_target,verbose=verbose)
    
    # Swap temperatures
    trajectories = replica.trajectories_.copy()
    E0 = trajectories['Energy'].iat[-1]
    B0 = replica.beta

    replica.beta = exchange(E0,B0)
    
    G = replica.graph_
    
    # Store trajectories
    if cycle == 0:
        trajectories_all = trajectories
    else:
        trajectories_all = pd.concat([trajectories_all,trajectories])
        
# Create directory if it doesn't exist
now = datetime.datetime.now()
time = now.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = os.path.join(DATA_DIR,'miniatures',graph_name,str(n_vertices),time)
if (rank == 0) and (not os.path.exists(output_dir)):
    os.makedirs(output_dir)
    
comm.Barrier()

# store your energy and rank
my_energy_core = np.array([(E0,rank)],dtype=[('energy',np.float64), ('rank',np.int32)])
# allocate memory for min energy and rank
min_energy_core = np.empty(1,dtype=[('energy',np.float64), ('rank',np.int32)])

# reduce to all ranks the mimum energy and rank that has the mimimum
comm.Allreduce([my_energy_core, 1, MPI.DOUBLE_INT], [min_energy_core, 1, MPI.DOUBLE_INT], op=MPI.MINLOC)

if rank == min_energy_core['rank']:
    file_name_adj = os.path.join(output_dir,'graph.gexf')
    nx.write_gexf(replica.graph_,file_name_adj)
    metrics_out = dict(trajectories_all.iloc[-1])
    
    # Store final metrics
    file_name_metrics_out = os.path.join(output_dir,f"metrics_mini.json")
    with open(file_name_metrics_out,'w+') as json_file:
        json.dump(metrics_out,json_file,indent=4)

# Store trajectories
file_name_trajectory = os.path.join(output_dir,f"replica_{rank}.csv")
trajectories_all.to_csv(file_name_trajectory,index=False,sep=',')




