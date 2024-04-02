#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

Daley-Kendall test code

dI/dt =
dS/dt =
dZ/dt =

Individual's states can have the following values:
0 : ignorant
1 : spreader
2 : zealot
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
if not os.path.exists('../results/'+directory_name+'DK/'):
    os.makedirs('../results/'+directory_name+'DK/',exist_ok=True)


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

# How many agents are initally spreaders
S0 = 1

num_i = N-S0
num_s = S0
num_z = 0

if S0>N:
    raise Exception('Initial infected agents greater than number of agents')

# instantiate the state for each agent as susceptible
states = np.zeros(N)

# randomly select S0 unique agents to be initially infected
random_initial_spreaders = np.random.choice(range(N),S0,replace=False)
# set the selected agents to infected
states[random_initial_spreaders] = 1
# create a copy of the states
temp_states = states.copy()

step = 0

# create an array to save the trajectory data
trajectory = np.zeros((MAX_ITERS+1,3))
# save the initial condition

## TODO generalize this so that user specifies S0,I0,R0
trajectory[0,0] = N-S0
trajectory[0,1] = S0
trajectory[0,2] = 0

# run the simulation until either the maximum steps have been reached or none of
# the agents are infected

# Iterate as long as you have not hit the maximum number of steps
while step < MAX_ITERS:

    # loop over infected agents
    for spreader_id in np.where(states == 1)[0]:
        # iterate over infected nodes neighbors
        for neighbor_id in neighbors[spreader_id]:
            # test for interaction
            if BETA > np.random.uniform(0,1):

                # if a spreader and ignorant interact ignorant becomes spreader
                if temp_states[neighbor_id] == 0:
                    temp_states[neighbor_id] = 1
                    num_i -= 1
                    num_s += 1
                # if a spreader and spreader interact both become zealots
                if temp_states[neighbor_id] == 1:
                    temp_states[spreader_id] = 2
                    temp_states[neighbor_id] = 2
                    num_s -= 2
                    num_z += 2
                # if a spreader and zealot interact spreader becomes zealot
                if temp_states[neighbor_id] == 2:
                    temp_states[spreader_id] = 2
                    num_s -= 1
                    num_z += 1

    # increment step
    step += 1
    # copy temp states to states
    states = temp_states.copy()
    trajectory[step,0] = num_i
    trajectory[step,1] = num_s
    trajectory[step,2] = num_z

    # Check if everyone has recovered and if so set rest of trajectory values to
    # last valid value.
    if num_s == 0:
        trajectory[step:] = trajectory[step]
        break

np.save('../results/'+directory_name+'DK/run_'+datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S_%f'),trajectory)
# # print(step)
# print(states)
# print(sys.argv[1]+'_sir')
# print(f'Rumor size: {len(np.where(states==2)[0])+len(np.where(states==1)[0])}')
