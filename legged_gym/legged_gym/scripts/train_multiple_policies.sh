#!/bin/bash


# Iteration number variable:
iter_num=3000
# train_num=3
num_envs=4096
python train.py --headless --max_iter=$iter_num --task="go1_upwards" --num_envs=$num_envs || python train.py --headless --max_iter=$iter_num --task="go1_upwards" --num_envs=$num_envs 

iter_num=10000
python train.py --headless --max_iter=$iter_num --task="go1_forward" --num_envs=$num_envs --resume || python train.py --headless --max_iter=$iter_num --task="go1_upwards" --num_envs=$num_envs --resume
