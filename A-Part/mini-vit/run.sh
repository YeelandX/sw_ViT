source ../swptenv.sh
source $SWPYTORCH_DIR/setenv-swpython
source $SWPYTORCH_DIR/setenv-swpytorch

export STASK_MALLOC_CROSS=1
bsub -I -p -akernel -b -q q_cpc -N 1 -cgsp 64 -host_stack 1024 -ro_size 256 -share_size 1600 -cross_size 60000 -mpecg 6 python3 $1
