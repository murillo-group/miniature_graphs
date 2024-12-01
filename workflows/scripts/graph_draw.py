import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import networkx as nx
from fa2 import ForceAtlas2
import sys
from scipy.sparse import load_npz



# Load the graph
adjacency_file = sys.argv[1]
drawing_file = sys.argv[2]
graph = nx.from_scipy_sparse_matrix(load_npz(adjacency_file))

# Calculate layout using force atlas
forceatlas2 = ForceAtlas2(gravity=1.0)

positions = forceatlas2.forceatlas2_networkx_layout(graph,
                                                    pos=None,
                                                    iterations=1000)

# Draw Graph
fig, ax = plt.subplots(figsize=(5,5),dpi=300)
nx.draw_networkx_nodes(graph,positions,node_size=2,node_color="red",alpha=0.4)
nx.draw_networkx_edges(graph,positions,edge_color="black",alpha=0.2)
plt.axis('off')

plt.savefig(drawing_file,bbox_inches='tight')