#!/bin/bash

# miniaturize a graph
echo mpiexec -n 2 python ../miniaturize/parallel_tempering_main.py example_graph/ graph.npz data.txt

read -rsn1 -p"Press any key to continue";echo

mpiexec -n 2 python ../miniaturize/parallel_tempering_main.py example_graph/ graph.npz data.txt

# Run sir simulation
echo python sir_main.py example_graph/ graph.npz

python ../simulations/sir_main.py example_graph/ graph.npz

echo python dk_main.py example_graph/ graph.npz

python ../simulations/dk_main.py example_graph/ graph.npz

echo python sir_analyze.py example_graph/ graph.npz

python ../analysis/sir_analyze.py example_graph/ graph.npz

echo python ../plot/sir_plot.py example_graph/

python ../plot/sir_plot.py example_graph/

echo python ../plot/dk_plot.py example_graph/

python ../plot/dk_plot.py example_graph/
