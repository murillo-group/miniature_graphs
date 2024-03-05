import networkx as nx
import numpy as np
from scipy.sparse import load_npz
import sys

MAX_ITERS = 100
BETA = .2
GAMMA = .2

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
I0 = 1
# 0 : dS/dt = -beta SI
# 1 : dI/dt = beta SI - gamma I
# 2 : dR/dt = gamma I
states = np.zeros(N)


random_initial_sick = np.random.choice(range(N),I0,replace=False)
states[random_initial_sick] = 1

temp_states = states.copy()

step = 0
while step < MAX_ITERS and states.sum()<N*2:

    # loop over infected agents
    for infected_id in np.where(states == 1)[0]:
        # iterate over infected nodes neighbors
        for neighbor_id in neighbors[infected_id]:
            # ensure the neighbor is susceptible and roll for infection
            if states[neighbor_id] == 0 and BETA > np.random.uniform(0,1):
                # switch neighbor to infected
                temp_states[neighbor_id] = 1
        # check for recovery
        if GAMMA > np.random.uniform(0,1):
            temp_states[infected_id] = 2



    # increment step
    step += 1
    # copy temp states to states
    states = temp_states.copy()

print(step)
print(states)
print(f'Epidemic size: {len(np.where(states==2)[0])+len(np.where(states==1)[0])}')
print(sys.argv[1]+'_sir')
