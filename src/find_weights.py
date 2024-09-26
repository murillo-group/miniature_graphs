
import networkx as nx\
from copy import deepcopy\
import matplotlib.pyplot as plt\

"""
FIXME: Include import from Metropolis Module
"""
    
target_density = 4.81e-4
target_assort = (-8.68e-2+1)/2
target_clustering = 1.59e-1


target_dict_const_weight = {'target_degree' : target_density,
                            'degree_weight' : 1.0,
                            'target_assort' : target_assort,
                            'assort_weight' : 1.0,
                            'target_clust' : target_clustering,
                            'clust_weight' : 1.0
                            }


estimates = 1
val = 50

assorts = np.zeros((estimates,val))
densitys = np.zeros((estimates,val))
clusts = np.zeros((estimates,val))
# energys = np.zeros((estimates,val))\

for est in range(estimates):
    obj = Metropolis(nx.erdos_renyi_graph(1681,.000481),0)
    
    for i in range(val):\

        assorts_init = (nx.degree_assortativity_coefficient(obj.get_graph())+1)/2
        densitys_init = nx.density(obj.get_graph())
        clusts_init = nx.average_clustering(obj.get_graph())

#         energy_init = obj.get_energy(**target_dict_const_weight)

        obj.run_step(target_dict_const_weight,num_changes=np.random.choice(range(10)))
        assorts[est,i] = abs(assorts_init - (nx.degree_assortativity_coefficient(obj.get_graph())+1)/2 )
        densitys[est,i] = abs(densitys_init - nx.density(obj.get_graph()))
        clusts[est,i] = abs(clusts_init - nx.average_clustering(obj.get_graph()))

#         energys[est,i] = energy_init-obj.get_energy(**target_dict_const_weight)


target_degree_weight = 1/np.mean(densitys)
target_assort_weight = 1/np.mean(assorts)
target_clust_weight = 1/np.mean(clusts)