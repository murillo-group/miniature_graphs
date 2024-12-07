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
from abc import ABC,abstractmethod
from collections import deque

NX_DENSITY = lambda G: nx.density(G)
NX_AVERAGE_CLUSTERING = lambda G: nx.average_clustering(G)
NX_DEGREE_ASSORTATIVITY = lambda G: nx.degree_assortativity_coefficient(G)

def sigmoid(x,x0,k):
    return 1 / (1 + np.exp(-k*(x-x0)))

def schedule_sigmoid(t_max,beta_max=1):
    k = 2*np.log(19)/t_max 
    t0 = t_max / 2
    
    return lambda t: sigmoid(t,t0,k) * beta_max

class Change(ABC):
    @abstractmethod
    def __init__(self,edges):
        pass 
    
    @abstractmethod
    def do(self,G):
        pass
    
    @abstractmethod
    def undo(self,G):
        pass
    
class Add(Change):
    def __init__(self,edge):
        self.edge = edge 
        
    def do(self,G):
        G.add_edge(self.edge[0],self.edge[1])
    
    def undo(self,G):
        G.remove_edge(self.edge[0],self.edge[1])
        
class Remove(Change):
    def __init__(self,edge):
        self.edge = edge 
        
    def do(self,G):
        G.remove_edge(self.edge[0],self.edge[1])
        
    def undo(self,G):
        G.add_edge(self.edge[0],self.edge[1])
        
class Switch(Change):
    def __init__(self,edges):
        self.edges = edges
    
    def do(self,G):
        old, new = self.edges
        G.remove_edge(old[0],old[1])
        G.add_edge(new[0],new[1])
        
    def undo(self,G):
        old, new = self.edges 
        G.remove_edge(new[0],new[1])
        G.add_edge(old[0],old[1])

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
                 metrics,
                 n_iterations: int,
                 weights=None,
                 n_changes: int = 1,
                 func_loss = None,
                 ):
        '''Instantiates the MH annealer
        '''
        # Store Annealing schedule
        self.schedule = schedule
        
        # Store metrics functions
        self._metrics = metrics 
    
        # Store weights
        if weights is None:
            # Weight metrics equally if no weights are specified
            self._weights = {key: 1.0 for key in self._metrics}
        else:
            self._weights = weights
        
        # Check for matching keys
        if self._metrics.keys() != self._weights.keys():
            raise ValueError("Specified weights don't match corresponding metrics")
        
        # Store number of changes per step
        self.n_iterations = n_iterations
        self.n_changes = n_changes
        
        # Store loss function
        if func_loss is None:
            self.func_loss = lambda weights, metrics_diff: np.sum(weights * np.abs(metrics_diff))
        else:
            self.func_loss = func_loss
        
    @property 
    def __state(self):
        state = np.zeros((self._n_states+2,))
            
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
        return list(self._metrics.keys())
    
    @property
    def weights(self):
        return list(self._weights.keys())
    
    @property
    def trajectories_(self):
        names = ['Beta','Energy'] + list(self._targets_names) 
        return pd.DataFrame(self._trajectories_,columns=names)
    
    def __get_metrics(self):
        '''Calculates the metrics of a graph
        '''
        metrics = []
        
        for name in self._targets_names:
            # Calculate metric
            metric = self._metrics[name](self.graph_)
            
            if not isinstance(metric, (int,float)):
                raise TypeError(f"Metric function {name} returned a non-scalar value")
            else:
                metrics.append(metric)
                
        return metrics
            
    def __energy(self, metrics):
        '''Energy of the graph with respect to target metrics
        '''
        diff = self._targets - metrics
        
        energy = self.func_loss(self._weights_arr,diff)
        
        return energy
        
    def __make_change(self) -> None:
        '''Implements changes in a graph
        '''
        self._actions = deque()
        
        # Propose changes
        for i in range(self.n_changes):
            # Propose change
            choose_action = True
            while choose_action:
                # Choose an action at random
                p = np.random.uniform()
                edges = list(nx.edges(self.graph_))
                non_edges = list(nx.non_edges(self.graph_))

                if (p < 0.25) and (len(non_edges) > 0):
                    # Add edge
                    idx = np.random.randint(0,len(non_edges))
                    
                    action = Add(non_edges[idx])
                    choose_action = False
                    
                elif (p < 0.5) and (len(edges) > 0):
                    # Remove edge
                    idx = np.random.randint(0,len(edges))
                    
                    action = Remove(edges[idx])
                    choose_action = False
                    
                elif (len(edges) > 0) and (len(non_edges) > 0):
                    # Switch edge
                    idx_edge = np.random.randint(0,len(edges))
                    idx_non_edge = np.random.randint(0,len(non_edges))
                
                    action = Switch((edges[idx_edge],non_edges[idx_non_edge]))
                    choose_action = False      
            
            # Implement change
            action.do(self.graph_)
            self._actions.append(action)          
    
    def __accept_change(self,E0: float, E1:float) -> bool:
        '''Accepts proposed change according to the Metropolis ratio
        '''
        return np.exp((E0-E1)*self.__beta) >= np.random.uniform()

            
    def transform(self, graph_seed, targets,verbose=False) -> None:
        '''Miniaturizes seed graph
        '''
        # Verify matching keys
        self._targets_names = set(self.metrics).intersection(set(targets.keys()))
        self._n_states = len(self._targets_names)
        if self._n_states != 0:
            # Initialize internal variables
            self._weights_arr = np.array([self._weights[key] for key in self._targets_names])
            self._targets = np.array([targets[key] for key in self._targets_names])
        else:
            raise LookupError(f"No valid targets specified for annealer with metrics {self.metrics}\n")

        # Initialize graph
        self.graph_ = deepcopy(graph_seed)
        
        # Calculate graph metrics and energy
        self.__m0 = self.__get_metrics()
        self.__E0 = self.__energy(self.__m0)
        
        # Initialize trajectories
        self._trajectories_ = np.zeros((self.n_iterations,
                                         self._n_states+2))
        
        # Iterate
        self.__step = 0
        while self.__step < self.n_iterations:
            if verbose is True:
                print(f"Iteration {self.__step+1}/{self.n_iterations}\n")
                
            # Change the graph and calculate energy
            self.__make_change()
            m1 = self.__get_metrics()
            E1 = self.__energy(m1)
            
            # Check for change
            if self.__accept_change(self.__E0, E1):
                # Update metrics and energy
                self.__m0 = m1
                self.__E0 = E1
            else:
                # Reverse changes
                for i in range(len(self._actions)):
                    self._actions.pop().undo(self.graph_)
            
            # Record state
            self._trajectories_[self.__step][0] = self.__beta
            self._trajectories_[self.__step][1] = self.__E0
            self._trajectories_[self.__step][2:] = self.__m0
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
        
        
        
        
        
            
        
        