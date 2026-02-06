#!/bin/bash


CUDA_VISIBLE_DEVICES=3 torchrun --nproc_per_node=1 --master_port=1703 basicsr/test.py -opt options/test_MD_RSID.yml