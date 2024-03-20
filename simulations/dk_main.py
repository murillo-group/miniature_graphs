#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated March 2024

Daley-Kendall test code
'''
import networkx as nx
import numpy as np
from scipy.sparse import load_npz
import sys


MAX_ITERS = 100
BETA = .2


# load the adjacency matrix for a graph
A = load_npz(sys.argv[1])

# number of agents in graph
N = A.shape[0]

neighbors = {}

# create a dictionary storing the neighbors of each rank
for i in range(N):
    neighbors[i] = A.getrow(i).indices

# delete the adjacency matrix to save some space
del A
# print(neighbors)

# How many agents are initally infected
S0 = 1
# 0 : I
# 1 : S
# 2 : Z
states = np.zeros(N)


random_initial_spreaders = np.random.choice(range(N),S0,replace=False)
states[random_initial_spreaders] = 1

temp_states = states.copy()

step = 0
while step < MAX_ITERS and states.sum()<N*2:

    # loop over infected agents
    for spreader_id in np.where(states == 1)[0]:
        # iterate over infected nodes neighbors
        for neighbor_id in neighbors[spreader_id]:
            # test for interaction
            if BETA > np.random.uniform(0,1):

                # if a spreader and ignorant interact ignorant becomes spreader
                if states[neighbor_id] == 0:
                    temp_states[neighbor_id] = 1
                # if a spreader and spreader interact both become zealots
                if states[neighbor_id] == 1:
                    temp_states[spreader_id] = 2
                    temp_states[neighbor_id] = 2
                # if a spreader and zealot interact spreader becomes zealot
                if states[neighbor_id] == 2:
                    temp_states[spreader_id] = 2

    # increment step
    step += 1
    # copy temp states to states
    states = temp_states.copy()


print(step)
print(states)
print(sys.argv[1]+'_sir')
print(f'Rumor size: {len(np.where(states==2)[0])+len(np.where(states==1)[0])}')
