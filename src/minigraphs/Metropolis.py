'''A module to miniaturize graphs using a Metropolis-Hastings annealer

    - Author: David J. Butts
    - Date created: March 2024
    - Date Last Modified: September 2024
'''

import numpy as np
import networkx as nx
import pandas as pd
from copy import deepcopy
from scipy.special import comb

class Metropolis():
    '''An MH-based annealer to miniaturize a graph
    
    Attritubes:
        - metrics_dict: A dictionary of function handles used by the annealer to
          calculate the functio metrics
          
        - beta: The initial inverse temperature of the replica
        
        - graph_: The generated miniature
    '''

    def __init__(self,
                 beta: float, 
                 metrics_funcs,
                 n_iterations: int,
                 metrics_weights=None,
                 n_changes: int = 1,
                 func_loss = None,
                 ):
        '''Instantiates the MH annealer
        '''
        # Store inverse temperature
        self.beta = beta
        
        # Store metrics functions
        self.metrics_funcs = metrics_funcs
        
        # Store metrics names
        self._metrics = metrics_funcs.keys()
        
        # Store number of iterations
        self.n_iterations = n_iterations
        
        # Store weights
        if metrics_weights is None:
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
        
        # Propose changes
        changes = np.random.randint(0,3,size=self.n_changes)
        
        for change in changes:
            if change == 0:
                # Add edge
                edges = list(nx.non_edges(temp_graph))
                idx = np.random.randint(0,len(edges))
                
                temp_graph.add_edge(*edges[idx])
                
            elif change == 1:
                # Remove edge
                edges = list(nx.edges(temp_graph))
                idx = np.random.randint(0,len(edges))
                
                temp_graph.remove_edge(*edges[idx])
                
            else:
                edges = list(nx.edges(temp_graph))
                non_edges = list(nx.non_edges(temp_graph))
                
                idx_edge = np.random.randint(0,len(edges))
                idx_non_edge = np.random.randint(0,len(non_edges))
                
                temp_graph.remove_edge(*edges[idx_edge])
                temp_graph.add_edge(*non_edges[idx_non_edge])
                
        return temp_graph
    
    def __accept_change(self,E0: float, E1:float) -> bool:
        '''Accepts proposed change according to the Metropolis ratio
        '''
        # Calculate Boltzmann Ratio 
        prob = np.exp((E0-E1)*self.beta)
    
        return prob >= np.random.uniform(0,1)

            
    def transform(self, graph_seed, metrics_target,verbose=False) -> None:
        '''Miniaturizes seed graph
        '''
        def step():
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
        
        def get_state():
            state = np.zeros((self.__n_states+2,))
            
            # Store current graph state
            state[0] = self.beta
            state[1] = self.__E0
            state[2:] = self.__m0
        
            return state

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
        for iter in range(self.n_iterations):
            if verbose is True:
                print(f"Iteration {iter+1}/{self.n_iterations}\n")
                
            step()
            self._trajectories__[iter] = get_state()