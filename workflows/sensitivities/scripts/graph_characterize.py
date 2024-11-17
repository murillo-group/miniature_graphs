from minigraphs.miniaturize import NX_ASSORTATIVITY, NX_CLUSTERING, NX_DENSITY
import networkx as nx 
from yaml import dump
from utils import load_graph

def graph_components(G):
    # Get connected components of the graph
    components = sorted(nx.connected_components(G), key=len, reverse=True)
    
    # Generate induced graphs
    graphs = [G.subgraph(c).copy() for c in components]

    return graphs

# PREAMBLE 
adjacency_file = snakemake.input[0]
metrics_file = snakemake.output[0]

functions = {
    'density': NX_DENSITY,
    'clustering': NX_CLUSTERING,
    'assortativity': NX_ASSORTATIVITY,
}

# LOAD GRAPH
graph = load_graph(adjacency_file)

# CALCULATE METRICS
components = graph_components(graph)

metrics = {metric: func(graph) for metric, func in functions.items()}
metrics['n_components'] = len(components)
metrics['connectivity'] = components[0].number_of_nodes() / graph.number_of_nodes()

# WRITE METRICS
with open(metrics_file,'w') as file:
    dump(metrics,file,default_flow_style=False)
    