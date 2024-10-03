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
    N = int(sys.argv[4])
    density = float(sys.argv[5])
    
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
params = []
for i in range(N):
    print(f"Sweep {i+1}/{N}")
    replica = Metropolis(0,
                         funcs_metrics,
                         n_iterations=n_iterations,
                        )
    
    G = nx.erdos_renyi_graph(n_vertices,density)
    replica.transform(G,metrics)

    df = replica.trajectories_.copy()

    for metric in replica.metrics:
        df.loc[:,metric] = df[metric] - metrics[metric]
        
    weights = dict(1/df[replica.metrics].abs().mean())

    print(f"Weights: {weights}")

    # Calculate optimal beta
    replica = Metropolis(0,
                        funcs_metrics,
                        n_iterations=n_iterations,
                        metrics_weights=weights,
                        )
    
    G = nx.erdos_renyi_graph(n_vertices,density)
    replica.transform(G,metrics)

    df = replica.trajectories_.copy()

    beta = -log(0.23) * 1/df['Energy'].diff().abs().mean()

    print(f"Beta: {beta}\n")

    params.append([beta] + list(weights.values()))

params = pd.DataFrame(params,columns=['beta'] + list(weights.keys()))
print(params)

params = params.mean()
print("Final parameters:")
print(params)



# Create outpot directory
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Save to JSON
file_name = os.path.join(output_dir,f'params_{n_vertices}.json')
with open(file_name,'w+') as f:
    json.dump(dict(params),f,indent=4)