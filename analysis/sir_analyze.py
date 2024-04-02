#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

Individual's states can have the following values:
0 : susceptible
1 : infected
2 : recovered

to run:

python .py example_graph/ graph.npz
'''
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys
import os

directory_name = sys.argv[1]
# graph_name = sys.argv[2]
#
# make directory if it hasn't been made yet
if not os.path.exists('../results/'+directory_name+'SIR/'):
    os.makedirs('../results/'+directory_name+'SIR/',exist_ok=True)

files = glob.glob('../results/'+directory_name+'/SIR/*')

all_data = np.zeros((len(files),3))

for i,file in enumerate(files):
    arr = np.load(file)
    all_data[i] = arr[-1]


# print(arr)

# arr = np.load('../simulations/Results/example_graph/graph.npz_SIR/run_2024_04_01_12_24_16_240273.npy')

# plt.show()
