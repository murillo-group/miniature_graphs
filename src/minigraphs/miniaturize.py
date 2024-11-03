'''A module to miniaturize graphs using a Metropolis-Hastings annealer

    - Author: David J. Butts
    - Date created: March 2024
    - Date Last Modified: September 2024
'''

import numpy as np
import networkx as nx
import pandas as pd
from copy import deepcopy
import scipy.sparse
from scipy.special import comb
import scipy
from sklearn.preprocessing import normalize
from typing import Callable 
import matplotlib.pyplot as plt

NX_DENSITY = lambda G: nx.density(G)
NX_AVERAGE_CLUSTERING = lambda G: nx.average_clustering(G)
NX_DEGREE_ASSORTATIVITY = lambda G: nx.degree_assortativity_coefficient(G)

def sigmoid(x,x0,k):
    return -1 / (1 + np.exp(-k*(x-x0))) + 1

def schedule_sigmoid(t_max,beta_max=1):
    k = 2*np.log(19)/t_max 
    t0 = t_max / 2
    
    return lambda t: sigmoid(t,t0,k) * beta_max

class MH:
    '''An MH-based annealer to miniaturize a graph
    
    Attritubes:
        - metrics_dict: A dictionary of function handles used by the annealer to
          calculate the functio metrics
          
        - beta: The initial inverse temperature of the replica
        
        - graph_: The generated miniature
    '''

    def __init__(self,
                 schedule: Callable, 
                 metrics_funcs,
                 n_iterations: int,
                 metrics_weights=None,
                 n_changes: int = 1,
                 func_loss = None,
                 ):
        '''Instantiates the MH annealer
        '''
        # Store Annealing schedule
        self.schedule = schedule
        
        # Store metrics functions
        self.metrics_funcs = metrics_funcs
        
        # Store metrics names
        self._metrics = metrics_funcs.keys()
        
        # Store number of iterations
        self.n_iterations = n_iterations
        
        # Store weights
        if metrics_weights is None:
            # Weight metrics equally if no weights are specified
            weights = {key: 1.0 for key in self._metrics}
        else:
            weights = metrics_weights
            
        self.metrics_weights = weights
        
        # Store number of changes per step
        self.n_changes = n_changes
        
        # Store loss function
        if func_loss is None:
            self.func_loss = lambda weights, metrics_diff: np.sum(weights * np.abs(metrics_diff))
        else:
            self.func_loss = func_loss
            
        # Update number of state variables
        self.__n_states = len(self._metrics)
        
    @property 
    def __state(self):
        state = np.zeros((self.__n_states+2,))
            
        # Store current graph state
        state[0] = self.__beta
        state[1] = self.__E0
        state[2:] = self.__m0
            
        return state

    @property
    def __beta(self):
        # Evaluate beta at the current time step
        return self.schedule(self.__step)
        
    @property
    def metrics(self):
        return list(self._metrics)
    
    @property
    def trajectories_(self):
        names = ['Beta','Energy'] + self.metrics 
        return pd.DataFrame(self._trajectories__,columns=names)
    
    def __get_metrics(self, graph):
        '''Calculates the metrics of a graph
        '''
        # Calculate graph metrics
        metrics = []
        
        for func_name, func in zip(self.metrics_funcs.keys(),self.metrics_funcs.values()):
            try:
                metric = func(graph)
                
                if not((type(metric) is int) or (type(metric) is float)):
                    raise(TypeError)
                
                metrics.append(metric)
                
            except TypeError:
                print("Metric produced non-scalar result: ")
                print(func_name)
                
        return metrics
            
    def __energy(self, metrics):
        '''Energy of the graph with respect to target metrics
        '''
        diff = self.__metrics_targets - metrics
        
        energy = self.func_loss(self.__weights,diff)
        
        return energy
        
    def __make_change(self):
        '''Implements changes in a graph
        '''
        
        # Make deep copy
        temp_graph = deepcopy(self.graph_)
        
        for i in range(self.n_changes):
            # Obtain list of current edges in the graph
            edges = list(nx.edges(temp_graph))
            non_edges = list(nx.non_edges(temp_graph))
            
            choose_action = True
            while choose_action:
                # Choose an action at random
                p = np.random.uniform()

                if (p < 0.25) and (len(non_edges) > 0):
                    # Add edge
                    idx = np.random.randint(0,len(non_edges))
                    
                    temp_graph.add_edge(*non_edges[idx])
                    choose_action = False
                    
                elif (p < 0.5) and (len(edges) > 0):
                    # Remove edge
                    idx = np.random.randint(0,len(edges))
                    
                    temp_graph.remove_edge(*edges[idx])
                    choose_action = False
                    
                elif (len(edges) > 0) and (len(non_edges) > 0):
                    # Switch edge
                    idx_edge = np.random.randint(0,len(edges))
                    idx_non_edge = np.random.randint(0,len(non_edges))
                    
                    temp_graph.remove_edge(*edges[idx_edge])
                    temp_graph.add_edge(*non_edges[idx_non_edge])
                    choose_action = False
                
        return temp_graph
    
    def __accept_change(self,E0: float, E1:float) -> bool:
        '''Accepts proposed change according to the Metropolis ratio
        '''
        return np.exp((E0-E1)*self.__beta) >= np.random.uniform()

            
    def transform(self, graph_seed, metrics_target,verbose=False) -> None:
        '''Miniaturizes seed graph
        '''
        # Verify matching keys for functions, weights, and metrics
        try:
            if self._metrics == metrics_target.keys() == self.metrics_weights.keys():
                
                # Initialize internal variables
                self.__weights = np.array([self.metrics_weights[key] for key in self._metrics])
                self.__metrics_targets = np.array([metrics_target[key] for key in self._metrics])
                
            else:
                raise(LookupError)
        
           
        except LookupError:
            print('Keys of functions, weights, or target metrics do not match.')
            
        # Initialize graph
        self.graph_ = graph_seed
        
        # Calculate graph metrics and energy
        self.__m0 = self.__get_metrics(self.graph_)
        self.__E0 = self.__energy(self.__m0)
        
        # Initialize trajectories
        self._trajectories__ = np.zeros((self.n_iterations,
                                         self.__n_states+2))
        
        # Iterate
        self.__step = 0
        while self.__step < self.n_iterations:
            if verbose is True:
                print(f"Iteration {self.__step+1}/{self.n_iterations}\n")
                
            # Change the graph
            graph_new = self.__make_change()
            
            # Retrieve energy of the current graph
            E0 = self.__E0

            # Calculate metrics and energy of the new graph
            m1 = self.__get_metrics(graph_new)
            E1 = self.__energy(m1)
            
            # Check for change
            if self.__accept_change(E0, E1):
                # Update current graph, metrics and energy
                self.graph_ = graph_new
                self.__m0 = m1
                self.__E0 = E1
                
            self._trajectories__[self.__step] = self.__state
            self.__step += 1
    
    @staticmethod
    def plot_trajectories(data,targets):
        trajectories = data.columns
        n_trajectories = len(trajectories)
        
        fig, axes = plt.subplots(n_trajectories,1,dpi=300,figsize=(5,n_trajectories))
        
        for i,trajectory in enumerate(trajectories):
            axes[i].plot(data[trajectory],linewidth=1.0)
            if (trajectory != 'Beta') and (trajectory != 'Energy'):
                axes[i].axhline(targets[trajectory],linestyle='--',linewidth=0.5,color='red')
            axes[i].set_ylabel(trajectory)
            
            if i != (n_trajectories-1):
                axes[i].set_xticklabels([])
                
        axes[n_trajectories-1].set_xlabel("Iteration")
            
            
class CoarseNET:
    '''Corsens an unweighted graph
    
    '''
    
    def __init__(self,alpha,G):
        '''Initializes coarsener 
        '''
        self.alpha = alpha
        self.G = deepcopy(G)
    
    @property
    def alpha(self):
        '''Shrinkage factor'''
        return self._alpha
    
    @alpha.setter
    def alpha(self,val):
        #TODO: Validate within range (0,1.0)
        self._alpha = val
        
    @property
    def G(self):
        '''Original Graph
        '''
        return self._G
    
    @G.setter
    def G(self,Graph):
        #TODO: Validate strongly connected graph
        self._G = Graph
    
    @staticmethod
    def adjacency(G):
        '''Computes the adjacency matrix of a Graph
        '''
        A = nx.to_scipy_sparse_array(G)
        A = A._asfptype()
        A = normalize(A,norm='l2',axis=0)
        
        return A
    
    @staticmethod
    def eigs(G):
        '''Computes the dominant eigenvalue and eigenvectors of the Adjacency matrix of a graph
        '''
        # Adjacency Matrix
        A = CoarseNET.adjacency(G)
        
        # Compute the first eigenvalue and right eigenvector
        lambda_, u_ = scipy.sparse.linalg.eigs(A,k=1)
        
        # Compute the left eigenvector
        _, v_= scipy.sparse.linalg.eigs(A.T,k=1)
                
        return np.real(lambda_)[0], np.real(np.squeeze(u_)), np.real(np.squeeze(v_))
    
    def __edge_score(self,edge):
        '''Calculates the score of a node pair
        '''
        u_a, u_b = self.u_[edge[0]], self.u_[edge[1]]
        v_a, v_b = self.v_[edge[0]], self.v_[edge[1]]
        
        prod = (self.lambda_-1)*(u_a+u_b)
        score = (-self.lambda_*(u_a*v_a+u_b*v_b) + v_a*prod + u_a*v_b + u_b*v_a) / (np.dot(self.v_,self.u_)-(u_a*v_a + u_b*v_b))  
    
        return score
        
    def __score(self):
        '''Calculates the score for all the edges in the graph
        '''
        # Initialize array of scores
        score = np.zeros(self.G_coarse_.number_of_edges())
        
        # Calculate score for every edge in the graph
        for i, edge in enumerate(self.G_coarse_.edges):
            score[i] = self.__edge_score(edge)
        
        return np.abs(score)
    
    def __contract(self,edge) -> bool:
        '''Updates graph by contracting nodes in the edge
        '''
        # Upack nodes
        u, v = edge
        left, right = self.nodes_coarse_[u], self.nodes_coarse_[v]
        
        contract = left != right
        if contract:
            # Merge nodes
            nx.contracted_nodes(self.G_coarse_,
                                left,
                                right,
                                self_loops=False,
                                copy=False)
            
            # Update node index in coarsened graph
            idx = self.nodes_coarse_ == right
            self.nodes_coarse_[idx] = left 
            self.nodes_removed_.append(right)
            
        return contract
        
    def coarsen(self) -> None:
        '''Coarsens the graph
        '''
        self.G_coarse_ = self._G.to_directed()
        n = self.G_coarse_.number_of_nodes()
        v = self.G_coarse_.number_of_edges()
        n_reduced = int(self._alpha * n)
        
        # Compute the eigenvalue and eigenvectors
        self.lambda_, self.u_, self.v_ = CoarseNET.eigs(self.G_coarse_)
        
        # Arrays of nodes and edges
        self.nodes_coarse_ = np.arange(0,n,dtype=np.int32)
        self.nodes_removed_ = []
        edges = list(self.G_coarse_.edges)
        
        # Calculate sorting indices according to score
        score = self.__score()
        idx = np.argsort(score)
        
        i = 0
        while i <= n_reduced:
            # Retrieve edge according to sorting
            edge = edges[idx[i]]
            
            # Contract edges
            contract = self.__contract(edge)
            
            i += 1
            
        # Add removed edges to the original graph
        self.G_coarse_.add_nodes_from(self.nodes_removed_)
        
        
        
        
        
            
        
        