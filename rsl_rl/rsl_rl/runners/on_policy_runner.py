# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import time
import os
from collections import deque
import statistics
import numpy as np
import copy

from torch.utils.tensorboard import SummaryWriter
import torch
import wandb

from rsl_rl.algorithms import PPO
from rsl_rl.modules import ActorCritic, ActorCriticRecurrent
from rsl_rl.env import VecEnv


class OnPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        if self.env.num_privileged_obs is not None:
            num_critic_obs = self.env.num_privileged_obs 
        else:
            num_critic_obs = self.env.num_obs
        actor_critic_class = eval(self.cfg["policy_class_name"]) # ActorCritic
        actor_critic: ActorCritic = actor_critic_class( self.env.num_obs,
                                                        num_critic_obs,
                                                        self.env.num_actions,
                                                        **self.policy_cfg).to(self.device)
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg: PPO = alg_class(actor_critic, device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]

        # init storage and model
        self.alg.init_storage(self.env.num_envs, self.num_steps_per_env, [self.env.num_obs], [self.env.num_privileged_obs], [self.env.num_actions])

        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        self.init_learning_iteration = 0

        _, _ = self.env.reset()
    
    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        # Total number of iterations (self.current_learning_iteration is the starting point)
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.init_learning_iteration = copy.deepcopy(self.current_learning_iteration)

        for it in range(self.init_learning_iteration, tot_iter):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
            
            # Update sigma reward curriculum:
            # if self.env.cfg.rewards.sigma_neg_rew_curriculum:
            #     self.env._update_reward_curriculum(it)

            mean_value_loss, mean_surrogate_loss, mean_entropy_loss = self.alg.update()
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
                
            # Every 500 iterations, update current_learning_iteration for retraining
            if it % 500 == 0:
                self.current_learning_iteration = it

            ep_infos.clear()
                    
        self.current_learning_iteration = self.init_learning_iteration + num_learning_iterations
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']
        wandb_dict = {}
        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                maxValue = self.env.reward_scales[key[4::]] / self.env.max_episode_length_s
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                wandb_dict['Episode_rew/' + key] = value / np.clip(np.abs(self.env.reward_scales[key[4::]]),1e-11,None)
                if key[4:8] == "task": # Only print max task rewards
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f} / {maxValue:.2f}\n"""
                else:
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""

        mean_std = self.alg.actor_critic.std.mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        max_episode_count = np.floor(self.env.common_step_counter / self.env.max_episode_length)
        max_latency = self.env.latency_range[1]
        if self.env.cfg.env.jump_type == "forward_with_obstacles":
            curriculum_level = torch.mean(self.env.terrain_levels.float())
        elif self.env.cfg.env.jump_type == "forward":
            curriculum_level = torch.mean(self.env.command_dist_levels.float())
        else:
            curriculum_level = torch.zeros(1)
        mean_max_jump = torch.median(self.env.max_height[torch.logical_and(~self.env._has_jumped_rand_envs,self.env.has_jumped)])
        mean_min_jump = torch.median(self.env.min_height[torch.logical_and(~self.env._has_jumped_rand_envs,self.env.has_jumped)])
        mean_task_pos_error = torch.mean(self.env.tracking_error_store)
        command_curriculum_level = self.env.command_curriculum_iter
        mean_base_acc = self.env.mean_base_acc_stored
        mean_dof_acc = self.env.mean_dof_acc_stored
        mean_action_rate = self.env.mean_action_rate_stored
        mean_termination_landing_level = torch.mean(self.env.reset_landing_error.float())
        mean_termination_landing_allowed = torch.mean(((self.env.reset_landing_error.clone() * self.env.cfg.env.reset_landing_error / self.env.cfg.commands.num_levels).clip(min=0.1)).float())
        memory_usage = self.env.memory_log
        mean_domain_rand_prob = torch.mean(self.env.pos_vel_randomisation_prob)

        if self.env.cfg.env.jump_type == "forward_with_obstacles":
            # terrain_curriculum_distribution = torch.bincount(self.env.terrain_levels,minlength=self.env.cfg.terrain.num_rows)/self.env.num_envs
            # terrain_curriculum_distribution_adjusted = torch.zeros(self.env.cfg.terrain.num_rows-self.env.cfg.terrain.num_zero_height_terrains+1)
            # terrain_curriculum_distribution_adjusted[0] = torch.sum(terrain_curriculum_distribution[0:self.env.cfg.terrain.num_zero_height_terrains])
            # terrain_curriculum_distribution_adjusted[1:] = terrain_curriculum_distribution[self.env.cfg.terrain.num_zero_height_terrains:]
            terrain_heights = self.env.env_properties[:,2]
            distribution_params = torch.tensor([torch.mean(terrain_heights), torch.std(terrain_heights)])
        elif self.env.cfg.env.jump_type == "forward":
            jump_dist = self.env.cfg.commands.ranges.pos_dx_ini[1] * self.env.command_dist_levels / self.env.cfg.commands.num_levels
            distribution_params = torch.tensor([torch.mean(jump_dist), torch.std(jump_dist)])
        else:
            terrain_curriculum_distribution_adjusted = torch.zeros(1)
            distribution_params = torch.zeros(2)
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Loss/entropy_loss', locs['mean_entropy_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        # self.writer.add_scalar('Perf/curriculum_level', curriculum_level, locs['it'])
        self.writer.add_scalar('Perf/mean_max_jump', mean_max_jump, locs['it'])
        self.writer.add_scalar('Perf/mean_task_pos_error', mean_task_pos_error, locs['it'])
        self.writer.add_scalar('Perf/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
        self.writer.add_scalar('Perf/curriculum mean', distribution_params[0], locs['it'])
        self.writer.add_scalar('Perf/curriculum std', distribution_params[1], locs['it'])
        self.writer.add_scalar('Perf/mean_base_acc', mean_base_acc, locs['it'])
        self.writer.add_scalar('Perf/mean_dof_acc', mean_dof_acc, locs['it'])
        self.writer.add_scalar('Perf/mean_action_rate', mean_action_rate, locs['it'])
        self.writer.add_scalar('Perf/memory_usage', memory_usage, locs['it'])
        self.writer.add_scalar('Perf/mean_termination_landing_level', mean_termination_landing_level, locs['it'])

        wandb_dict['Loss/value_function'] = locs['mean_value_loss']
        wandb_dict['Loss/surrogate'] = locs['mean_surrogate_loss']
        wandb_dict['Loss/entropy_loss'] = locs['mean_entropy_loss']
        wandb_dict['Loss/learning_rate'] = self.alg.learning_rate
        wandb_dict['Policy/mean_noise_std'] = mean_std.item()
        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        wandb_dict['Perf/curriculum_level'] = curriculum_level
        wandb_dict['Perf/sigma_rew_neg'] = self.env.sigma_rew_neg
        wandb_dict['Perf/mean_max_jump'] = mean_max_jump
        wandb_dict['Perf/mean_min_jump'] = mean_min_jump
        wandb_dict['Perf/mean_task_pos_error'] = mean_task_pos_error
        # wandb_dict['Perf/mean_reward'] = statistics.mean(locs['rewbuffer'])
        wandb_dict['Perf/curriculum mean'] = distribution_params[0]
        wandb_dict['Perf/curriculum std'] = distribution_params[1]
        wandb_dict['Perf/mean_base_acc'] = mean_base_acc
        wandb_dict['Perf/mean_dof_acc'] = mean_dof_acc
        wandb_dict['Perf/mean_action_rate'] = mean_action_rate
        wandb_dict['Perf/memory_usage'] = memory_usage
        wandb_dict['Perf/mean_termination_landing_level'] = mean_termination_landing_level
        wandb_dict['Perf/mean_domain_rand_prob'] = mean_domain_rand_prob



        if len(locs['rewbuffer']) > 0:
           wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
           wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])

           
        wandb.log(wandb_dict, step=locs['it'])
        str = f" \033[1m Learning iteration {locs['it']}/{self.init_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Max episodes elapsed:':>{pad}} {max_episode_count:.2f}\n"""
                          f"""{'Max latency:':>{pad}} {max_latency:.2f}\n"""
                          f"""{'Current neg sigma:':>{pad}} {self.env.sigma_rew_neg:.2f}\n"""
                        #   f"""{'Current command curriculum range:':>{pad}} {command_cur_range[0]:.2f} ', '{command_cur_range[1]:.2f} \n"""
                          f"""{'Current terrain curriculum distribution:':>{pad}} {distribution_params}\n"""
                         f"""{'Current terrain curriculum level:':>{pad}} {curriculum_level}\n"""
                          f"""{'Median max height:':>{pad}} {mean_max_jump:.2f}\n"""
                          f"""{'Median min height:':>{pad}} {mean_min_jump:.2f}\n"""
                          f"""{'Mean landing error:':>{pad}} {mean_task_pos_error:.2f}\n"""
                          f"""{'Mean max termination landing error :':>{pad}} {mean_termination_landing_allowed:.2f}\n""")

                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
                        #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                        #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        
        # Shift the iteration counter when training from a pre-trained model:
        shifted_it = locs['it'] - self.init_learning_iteration
        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{str.center(width, ' ')}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (shifted_it + 1) * (
                               locs['num_learning_iterations'] - shifted_it):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None):
        torch.save({
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }, path)

    def load(self, path, load_optimizer=True):
        loaded_dict = torch.load(path)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = loaded_dict['iter']
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
