from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch

def deploy(args):
    args.load_run = "Aug03_11-09-18_"
    # args.sim_device = "cpu"
    # args.checkpoint = 9000
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.env.jump_type = "forward"
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device='cpu')

    path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, path,args.load_run)
    print('Exported policy as jit script to: ', path)

if __name__ == '__main__':
    args = get_args()
    deploy(args)
