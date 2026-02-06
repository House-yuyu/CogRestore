#!/bin/bash

# export CUDA_VISIBLE_DEVICES=2,3


CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 --master_port=2302 basicsr/train.py -opt options/train_MD_RSID.yml
