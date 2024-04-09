#!/usr/bin/env python3
# -*- coding: utf8 -*-
import networkx as nx
import numpy as np
import sys

from miniaturize import my_caveman, profile

csvfile=open(sys.argv[1],"w",newline='')

# Decorate function
er = profile(csvfile)(nx.erdos_renyi_graph)
ba = profile(csvfile)(nx.barabasi_albert_graph)
ws = profile(csvfile)(nx.watts_strogatz_graph)
caveman = profile(csvfile)(my_caveman)

for i in range(0,3):
    for size in (2000,):
        # Time ER
        for p in np.arange(0.1,1,0.1):
            er(size,p)
        
        # Time BA
        for m in (size * np.arange(0.1,1,0.1)):
            ba(size,int(m))

        # Time WS
        for k in (size * np.arange(0.1,1,0.1)):
            for p in np.arange(0,1,0.1):
                ws(size,int(k),p)
        
        # Time Caveman
        for l in (2,5,10):
            caveman(size,int(l))






