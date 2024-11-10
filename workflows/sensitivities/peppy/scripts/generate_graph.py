import networkx as nx
from scipy.sparse import save_npz
import numpy as np


# Global configuration
file = snakemake.output.filename
n_vertices = snakemake.params.n_vertices
param = snakemake.params.parameter

# Generate graph
G = nx.erdos_renyi_graph(n_vertices,param)

# Save file
save_npz(file,nx.to_scipy_sparse_array(G))