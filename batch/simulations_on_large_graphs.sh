#!/bin/bash

DIR=$1
GRAPH=$2
SIMS=$3

# for s in $(seq 1 $SIMS)
# do
#   echo ../simulations/sir_main.py $DIR $GRAPH.npz
#   python ../simulations/sir_main.py $DIR $GRAPH.npz
# done


echo python sir_analyze.py $DIR $GRAPH.npz

python ../analysis/sir_analyze.py $1


echo python ../plot/sir_plot.py $1

python ../plot/sir_plot.py $1
