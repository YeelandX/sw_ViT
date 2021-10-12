#!/bin/sh
set -e

source ../swptenv.sh
source $SWPYTORCH_DIR/setenv-swpython
source $SWPYTORCH_DIR/setenv-swpytorch

cd swops && make clean && make && cd ..

export CC=./gcc
export CXX=./gcc

set -x

source ./compile_cmd.sh

bsub -I -p -akernel -health 2 -b -q q_cpc -n 1 -cgsp 64 -mpecg 6 -host_stack 1024 -ro_size 256 -share_size 11500 python3 setup.py install --user
