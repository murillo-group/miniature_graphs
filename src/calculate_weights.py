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
    n_vertices = int(sys.argv[2])
    n_iterations = int(sys.argv[3])
except IndexError:
    print("Error: not enough input arguments provided. Necessary inputs are\n")
    print("\t - Graph name\n\t - n_vertices\n\t - n_iterations\n")

# Retrieve Graph Density
DATA_DIR = os.environ['DATA_DIR']
input_file = os.path.join(DATA_DIR,'networks',graph_name,'metrics.json')
output_dir = os.path.join(DATA_DIR,'weights')

with open(input_file) as file:
    data = json.load(file)
    
density = data['density']

# Specify Metric funcions
funcs_metrics = {
    'density': nx.density,
    'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
    'clustering': nx.average_clustering
}

# Specify some metrics to 'match'
metrics = {
    'density' : 0,
    'assortativity_norm': 0,
    'clustering': 0
}

# Explore the space
replica = Metropolis(0,funcs_metrics,n_iterations,n_changes=10)
G = nx.erdos_renyi_graph(n_vertices,density)

replica.transform(G,metrics,verbose=True)

# Obtain the trajectories
df = replica.trajectories_
diff = df[list(funcs_metrics.keys())].diff().abs()

# Calculate the weights
weights = dict(1/diff.mean())
weights['n_vertices'] = n_vertices

print(weights)

# Create outpot directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to JSON
file_name = os.path.join(output_dir,graph_name,f'weights_{n_vertices}.json')
with open(file_name,'w+') as f:
    json.dump(weights,f,indent=4)