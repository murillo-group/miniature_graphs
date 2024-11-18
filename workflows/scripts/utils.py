from scipy.sparse import load_npz, save_npz
import networkx as nx

def save_graph(file,G):
    '''Saves the adjacency matrix of a graph'''
    save_npz(file,nx.to_scipy_sparse_array(G))

def load_graph(file):
    '''Loads the adjacency matrix of a graph'''
    return nx.from_scipy_sparse_array(load_npz(file))