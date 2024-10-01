from minigraphs import Metropolis
from numpy import log
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

# Retrieve Graph Metrics
DATA_DIR = os.environ['DATA_DIR']
input_file = os.path.join(DATA_DIR,'networks',graph_name,'metrics.json')
output_dir = os.path.join(DATA_DIR,'params',graph_name)

with open(input_file) as file:
    metrics = json.load(file)
    metrics = {key:metrics[key] for key in ['density','assortativity_norm','clustering']}

# Specify Metric funcions
funcs_metrics = {
    'density': nx.density,
    'assortativity_norm': lambda G: (nx.degree_assortativity_coefficient(G)+1)/2,
    'clustering': nx.average_clustering
}

# Calculate weights
replica = Metropolis(0,funcs_metrics,n_iterations,n_changes=10)
G = nx.erdos_renyi_graph(n_vertices,metrics['density'])
replica.transform(G,metrics,verbose=True)

df = replica.trajectories_

for metric in replica.metrics:
    df.loc[:,metric] = df[metric] - metrics[metric]
    
weights = dict(1/df[replica.metrics].abs().mean())

print(f"Weights: {weights}")

# Calculate optimal beta
replica = Metropolis(0,
                     funcs_metrics,
                     metrics_weights=weights,
                     n_iterations=n_iterations,
                     n_changes=10
                     )
G = nx.erdos_renyi_graph(n_vertices,metrics['density'])
replica.transform(G,metrics,verbose=True)

df = replica.trajectories_

beta = -log(0.23) * 1/df['Energy'].diff().abs().mean()

print(f"Beta: {beta}")

params = {"beta": beta, "n_vertices":n_vertices, "weights": weights}

# Create outpot directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to JSON
file_name = os.path.join(output_dir,f'params_{n_vertices}.json')
with open(file_name,'w+') as f:
    json.dump(params,f,indent=4)