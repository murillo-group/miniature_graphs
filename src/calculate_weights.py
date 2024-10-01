from minigraphs import Metropolis
import pandas as pd 
import networkx as nx
import sys
import os
import json
'''Calculates the weights for the specified graph
'''

# Validate inputs
try:
    graph_name = sys.argv[1]
    n_vertices = sys.argv[2]
    n_iterations = sys.argv[3]
except IndexError:
    print("Error: not enough input arguments provided. Necessary inputs are\n")
    print("\t - Graph name\n\t - n_vertices\n\t - n_iterations\n")

# Retrieve Graph Density
DATA_DIR = os.getenv['DATA_DIR']
file_name = os.path.join(DATA_DIR,'networks','metrics.json')

with open(filename) as file:
    data = json.load(file)
    
density = data['density']

# Specify Metric funcions
funcs_metrics = {
    'density': nx.density,
    'assortativity_norm': lambda G: nx.degree_assortativity_coefficient(G)/2,
    'clustering': nx.average_clustering
}

# Specify some metrics to 'match'
metrics = {
    'density' : 0,
    'assortativity': 0,
    'clustering': 0
}

# Explore the space
replica = Metropolis(0,funcs_metrics,n_iterations)
G = nx.erdos_renyi_graph(N,density)

replica.transform(G,metrics,verbose=True)

# Obtain the trajectories
df = replica.trajectories_
diff = df[['density','assortativity','clustering']].diff().abs()

# Calculate the weights
weights = dict(1/diff.mean())
weights['num_vertices'] = N

print(weights)

# Save to JSON
with open('weights.json','w+') as f:
    json.dump(weights,f,indent=4)