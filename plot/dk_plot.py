#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@author: buttsdav@msu.edu
last updated April 2024

DK plotting code
'''
import matplotlib.pyplot as plt
import numpy as np
import glob
import sys

directory_name = sys.argv[1]
# graph_name = sys.argv[2]

# plot of all runs on a single plot
for i,file in enumerate(glob.glob('../results/'+directory_name+'/DK/*')):
    arr = np.load(file)
    plt.plot(arr[:,0],c='C0',alpha=.5,label='I')
    plt.plot(arr[:,1],c='C1',alpha=.5,label='S')
    plt.plot(arr[:,2],c='C2',alpha=.5,label='Z')
    if i == 0:
        plt.legend()
plt.xlabel('iterations',fontsize=14)
plt.ylabel('count',fontsize=14)
plt.show()
