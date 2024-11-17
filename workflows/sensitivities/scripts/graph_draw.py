import matplotlib as plt 
from utils import load_graph
import graph_tool.all as gt
import matplotlib.pyplot as plt 

# Load the graph
adjacency_file = snakemake.input[0]
drawing_file = snakemake.output[0]
graph = load_graph(adjacency_file)

# Get the edges
edges = list(graph.edges)

# Construct graph-tools Graph
H = gt.Graph(edges,directed=False)

# Plot graph
fig, ax = plt.subplots(figsize=(3,3),dpi=300)
artist = gt.graph_draw(H,mplfig=ax)
artist.fit_view(yflip=True)

plt.savefig(drawing_file,bbox_inches='tight')