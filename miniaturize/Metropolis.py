import numpy as np
import networkx as nx
from copy import deepcopy

class Metropolis():

    def __init__(self, initial_graph, beta):
        self.beta = beta
        self.graph = deepcopy(initial_graph)
        self.step = 0

    def make_change(self, num_changes=1):
        '''
        This function implements a change in a graph\
        '''
        # make a deepcopy of the graph to ensure changes to temp_graph are independent of graph
        temp_graph = deepcopy(self.graph)
        
        # loop through the number of change you want to make
        # note that it is possible that a change could be undone
        # e.g. change one adds an edge then change two adds it back
        for changes in range(num_changes):
            non_edges = list(nx.non_edges(temp_graph))
            edges = list(nx.edges(temp_graph))

            # select a random edge that does not exist in the graph
            new_node_one, new_node_two = non_edges[np.random.randint(len(non_edges))]

            # select a random edge that exists in the graph
            old_node_one, old_node_two = edges[np.random.randint(len(edges))]

            # move an edge half of the time
            if np.random.uniform(0,1) < .5:

                # add the non-existant edge and remove the old edge. This is equal to moving an edge\
                temp_graph.add_edge(new_node_one, new_node_two)
                temp_graph.remove_edge(old_node_one, old_node_two)

            # add or remove an edge the other half of the time\
            else:
                # add and edge half of the time
                if np.random.uniform(0,1) < .5:

                    # add a non-existant edge to the graph
                    temp_graph.add_edge(new_node_one, new_node_two)

                # remove an edge the other half of the time
                else:

                    # remove an edge from the graph
                    temp_graph.remove_edge(old_node_one, old_node_two)

        return temp_graph

    def energy_fnc(self, graph, **kwargs):
        '''
        This function defines the loss function for the metropolis algorithm
        '''
        graph_degree = nx.density(graph)
        graph_assort = nx.degree_assortativity_coefficient(graph)
        graph_clust = nx.average_clustering(graph)
        
        # max-min normalization\
        graph_assort = (graph_assort+1)/2\
        

#         energy = kwargs['degree_weight']*((graph_degree_normalized - kwargs['target_degree'])/kwargs['target_degree'])**2 + kwargs['assort_weight']*((graph_assort_normalized - kwargs['target_assort'])/kwargs['target_assort'])**2 + kwargs['clust_weight']*((graph_clust_normalized - kwargs['target_clust'])/kwargs['target_clust'])**2
        energy = kwargs['degree_weight']*abs((graph_degree - kwargs['target_degree'])) + \
                 kwargs['assort_weight']*abs((graph_assort - kwargs['target_assort'])) + \
                 kwargs['clust_weight']*abs((graph_clust - kwargs['target_clust']))

        return energy

    def metropolis_check(self, energy_before, energy_after):
        '''
        This function decides whether a change is kept or discarded based on the losses. Here, loss_a is the the
        '''
        # This checks if the change should be implemented, and returns True if so
        if np.exp((energy_before-energy_after)*self.beta) > np.random.uniform(0,1):
            return True
        # if the if statement misses, return False
        return False

    def run_step(self,targets_and_weights,num_changes=1):

        # change the graph
        changed_graph = self.make_change(num_changes)

        # check the energy of the current graph
        current_energy = self.energy_fnc(self.graph, **targets_and_weights)
        # print(current_energy)

        # check the energy of the chagned graph
        changed_energy = self.energy_fnc(changed_graph, **targets_and_weights)
        # print(changed_energy)

        # up the metropolis creteria to decide if the change should be made
        if self.metropolis_check(current_energy, changed_energy):
            # deepcopy the changed graph into the current graph
            self.graph = deepcopy(changed_graph)

            # if not self.step % 1:
                # print(f'Updated graph metrics\\nAve. Degree: \{2*self.graph.number_of_edges()/self.graph.number_of_nodes()\}, Assortativity: \{nx.degree_assortativity_coefficient(self.graph)\}, Clustering: {nx.average_clustering(self.graph)\}\\n')


        self.step += 1
        # update beta
        # self.beta = beta_update(beta,step)

    def get_graph(self):
        '''This function returns the graph'''

        return self.graph

    def get_energy(self, **kwargs):

        return self.energy_fnc(self.graph, **kwargs)
