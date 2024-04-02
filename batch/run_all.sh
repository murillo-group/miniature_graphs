#!/bin/bash

# Run sir simulation
echo python sir_main.py example_graph/ graph.npz

python ../simulations/sir_main.py example_graph/ graph.npz

echo python dk_main.py example_graph/ graph.npz

python ../simulations/dk_main.py example_graph/ graph.npz

echo python sir_analyze.py example_graph/ graph.npz

python ../analysis/sir_analyze.py example_graph/ graph.npz

echo python ../plot/sir_plot.py

python ../plot/sir_plot.py
