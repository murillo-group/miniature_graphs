#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

The following code simulates a compartmental SIR model that obeys the following
equations:

dS/dt = -beta SI
dI/dt = beta SI - gamma I
dR/dt = gamma I

Individual's states can have the following values:
0 : susceptible
1 : infected
2 : recovered

to run:

python sir_main.py example_graph/ graph.npz
'''

import networkx as nx
import numpy as np
from scipy.sparse import load_npz
import sys
import datetime
import os

##############
# Load graph #
##############

directory_name = sys.argv[1]
graph_name = sys.argv[2]

# make directory if it hasn't been made yet
if not os.path.exists('../results/'+directory_name+'SIR/'):
    os.makedirs('../results/'+directory_name+'SIR/',exist_ok=True)


# load the adjacency matrix for graph that the model will run on
# it is assumed that the graph will be an scipy.sparse matrix
A = load_npz('../data/'+directory_name+graph_name)

# find the number of agents in the graph
N = A.shape[0]

####################
# Parameters Setup #
####################

# maximum allowed iterations
MAX_ITERS = 200
# infection parameter for SIR model (these depend on the number of agents)
BETA = 20/N
# recovery parameter for SIR model (these depend on the number of agents)
GAMMA = 20/N

# create a dictionary storing the neighbors of each rank
neighbors = {}
for i in range(N):
    neighbors[i] = A.getrow(i).indices

# delete the adjacency matrix to save some space
del A

##################
# Initialization #
##################

# How many agents are initally infected
I0 = 1

num_s = N-I0
num_i = I0
num_r = 0

if I0>N:
    raise Exception('Initial infected agents greater than number of agents')

# instantiate the state for each agent as susceptible
states = np.zeros(N)

# randomly select I0 unique agents to be initially infected
random_initial_sick = np.random.choice(range(N),I0,replace=False)
# set the selected agents to infected
states[random_initial_sick] = 1
# create a copy of the states
temp_states = states.copy()

step = 0

# create an array to save the trajectory data
trajectory = np.zeros((MAX_ITERS+1,3))
# save the initial condition

## TODO generalize this so that user specifies S0,I0,R0
trajectory[0,0] = N-I0
trajectory[0,1] = I0
trajectory[0,2] = 0

# run the simulation until either the maximum steps have been reached or none of
# the agents are infected

# Iterate as long as you have not hit the maximum number of steps
while step < MAX_ITERS:

    # loop over infected agents
    for infected_id in np.where(states == 1)[0]:
        # iterate over infected nodes neighbors
        for neighbor_id in neighbors[infected_id]:
            # ensure the neighbor is susceptible and roll for infection
            if temp_states[neighbor_id] == 0 and BETA > np.random.uniform(0,1):
                # switch neighbor to infected
                temp_states[neighbor_id] = 1
                num_i += 1
                num_s -= 1
        # check for recovery
        if GAMMA > np.random.uniform(0,1):
            temp_states[infected_id] = 2
            num_i -= 1
            num_r += 1



    # increment step
    step += 1
    # copy temp states to states
    states = temp_states.copy()
    trajectory[step,0] = num_s
    trajectory[step,1] = num_i
    trajectory[step,2] = num_r

    # Check if everyone has recovered and if so set rest of trajectory values to
    # last valid value.
    if num_i == 0:
        trajectory[step:] = trajectory[step]
        break

np.save('../results/'+directory_name+'SIR/run_'+datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S_%f'),trajectory)
# print(trajectory,np.sum(trajectory,axis=1))
# print(sys.argv[1]+'_sir')
