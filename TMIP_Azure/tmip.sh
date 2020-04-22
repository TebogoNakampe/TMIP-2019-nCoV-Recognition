#!/bin/bash
source /opt/intel/inteloneapi/setvars.sh  > /dev/null 2>&1
source activate tensorflow
python src/train.py

