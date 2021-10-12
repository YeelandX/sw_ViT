#!/bin/sh
###################
# run this script like the followings:
# ./pip.sh list
# ./pip.sh show swextension
###################
set -e

SHARE_DIR=/home/export/online1/share/wxsc
source ${SHARE_DIR}/swpt/setenv-swpython

bsub -I -akernel -q q_cpc -n 1  pip3 $@

