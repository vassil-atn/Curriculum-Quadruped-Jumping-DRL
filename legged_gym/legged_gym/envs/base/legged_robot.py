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

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import gc
from scipy.spatial.transform import Rotation as R
from scipy import signal
# from scipy.spatial.transform import Slerp
import psutil
from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
from legged_gym.utils.butterworth import ButterworthFilter
from legged_gym.utils.EMA import EMAFilter
import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float,quat_slerp
from legged_gym.utils.math import quat_distance,torch_rand_float_tensor
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()
        self.gym.render_all_camera_sensors(self.sim)
        # Filter the actions:
        if self.cfg.control.use_action_filter:
            actions_filtered = self._filter_actions(self.actions)
        else:
            actions_filtered = self.actions

        self.actions_filtered = actions_filtered.clone()
        self.memory_log = psutil.virtual_memory()[3]/1000000000

        for _ in range(self.cfg.control.decimation):
            self._store_PD_states()
            self.torques = self._compute_torques(actions_filtered).view(self.torques.shape)
            self.torques_to_apply = self.torques.clone()
            if self.cfg.env.use_springs: # If using springs, apply their torques to the system
                self.torques_springs = self._compute_spring_torques()
                self.torques_to_apply += self.torques_springs.clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques_to_apply))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            if self.cfg.env.throttle_to_real_time:
                self.gym.sync_frame_time(self.sim)
        self.post_physics_step()

        if not self.headless:

            ref_env = self.cfg.viewer.ref_env

            # pos = np.array([self.cfg.viewer.pos]).copy().flatten()
            # pos[:2] = self.initial_root_states_nonrandomised[ref_env,:2].cpu().numpy() - np.array([0.,0.9])

            # lookat = np.array([self.cfg.viewer.lookat]).copy().flatten()
            # lookat[:2] = self.initial_root_states_nonrandomised[ref_env,:2].cpu().numpy() 
            
            pos_camera = self.env_origins[0].cpu().numpy() + np.array([0.35,-1.,0.32])
            lookat_camera =  self.env_origins[0].cpu().numpy() + np.array([0.,0.,0.32])


            pos = pos_camera
            lookat = lookat_camera

            if self.cfg.viewer.simulate_camera:
                self.gym.set_camera_location(self.camera_handle, self.envs[0],
                                    gymapi.Vec3(pos_camera[0],pos_camera[1],pos_camera[2]),
                                gymapi.Vec3(lookat_camera[0],lookat_camera[1],lookat_camera[2]))   
            
            if self.cfg.viewer.camera_track_robot:

                self.set_camera(pos.tolist(), lookat.tolist())

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.feet_pos = self.rigid_body_state[:,self.feet_indices,0:3]
        self.feet_vel = self.rigid_body_state[:,self.feet_indices,7:10]
        self.feet_acc = (self.feet_vel - self.last_feet_vel) / self.dt
        self.contacts = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.ori_error = quat_distance(self.base_quat[:], self.commands[:, 3:7])
        # self.error_quat = quat_distance(self.base_quat[:], self.commands[:, 3:7],as_quat=True)
        # self.ori_error = self._compute_yaw_error(self.root_states[:, 3:7], self.commands[:, 3:7])
        # _ = self._compute_yaw_error(self.root_states[:, 3:7], self.commands[:, 3:7])
        self.dof_acc = (self.dof_vel - self.last_dof_vel) / self.dt
        self.dof_jerk = (self.dof_acc - self.last_dof_acc) / self.dt      
        self.base_acc = (self.root_states[:,7:10] - self.last_root_vel[:,:3]) / self.dt
        self.imu_state = self.imu_tensor / (self.total_mass + self.payload_masses).unsqueeze(-1)
        # Filter out normal forces that are negative:
        self.force_sensor_readings[:, :, 2] = torch.clamp(self.force_sensor_readings[:, :, 2], min=0.)

        self.base_lin_vel_imu += quat_rotate_inverse(self.base_quat,self.imu_state[:, 0:3]) * self.dt 
        
        # self.action_rate_stored = torch.cat((self.action_rate_stored,((self.actions - self.last_actions) / self.dt).unsqueeze(0)),dim=0)
        # self.base_acc_stored = torch.cat((self.base_acc_stored,((self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt).unsqueeze(0)),dim=0)
        # self.dof_acc_stored = torch.cat((self.dof_acc_stored,self.dof_acc.unsqueeze(0)),dim=0)

        self.mean_action_rate_stored = torch.mean(torch.abs((self.actions[:] - self.last_actions[:]) / self.dt))
        self.mean_action_rate_2_stored = torch.mean(torch.abs((self.actions - 2*self.last_actions + self.last_last_actions) / (self.dt**2)))

        self.mean_base_acc_stored = torch.mean(torch.abs((self.root_states[:, 7:10] - self.last_root_vel[:, :3]) / self.dt))
        self.mean_dof_acc_stored = torch.mean(torch.abs(self.dof_acc[:]))

        if self.cfg.env.pass_remaining_time:
            self.remaining_time = self.max_episode_length - self.episode_length_buf

        self._post_physics_step_callback()


        # compute observations, rewards, resets, ...
        self.check_jump()
        self.check_termination()
        self.compute_reward()
        
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.num_steps_before_termination[env_ids] = self.episode_length_buf[env_ids]  
        
        if self.cfg.domain_rand.curriculum:
            self._update_domain_rand_curriculum(env_ids)
            # pass
        if self.cfg.commands.curriculum and self.jump_type == "forward":
            # pass
            self._update_command_curriculum(env_ids)
        # if self.cfg.rewards.sigma_neg_rew_curriculum:
        #     self._update_reward_curriculum()

        self.reset_idx(env_ids)


        # # For some of those envs that have started with has_jumped flag, change the flag 
        # # back to 0 to make them jump.
        if self.cfg.domain_rand.randomize_has_jumped and self.cfg.domain_rand.reset_has_jumped:
            rand_envs = self._has_jumped_rand_envs
            idx = torch.nonzero(torch.logical_and(rand_envs,self._reset_randomised_has_jumped_timer == self.episode_length_buf),as_tuple=False).flatten()
            if not self.cfg.domain_rand.manual_has_jumped_reset_time == 0:
                idx = torch.nonzero(torch.logical_and(rand_envs,self.cfg.domain_rand.manual_has_jumped_reset_time == self.episode_length_buf),as_tuple=False).flatten()
            self.has_jumped[idx] = False
            self._has_jumped_rand_envs[idx] = False
            self._reset_randomised_has_jumped_timer[idx] = 0
            # Keep track of when the has_jumped flag was switched back to 0 (for certain terminations):
            self._has_jumped_switched_time[idx] = self.episode_length_buf[idx]
        
        self._compute_state_history()

        self._store_states()

        idx,sampled_latency = self._sample_latency()

        self._get_delayed_states(idx,sampled_latency)

        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)

        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_dof_acc[:] = self.dof_acc[:]
        self.last_feet_vel[:] = self.feet_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]
        self.last_contacts[:] = self.contacts[:]
        self.base_acc_prev[:] = self.base_acc[:]

        roll,pitch,yaw = get_euler_xyz(self.base_quat)
        self.euler[:] = torch.stack((wrap_to_pi(roll),wrap_to_pi(pitch),wrap_to_pi(yaw)),dim=1)

        # Only update the max height achieved during the episode during the first jump while in mid-air:
        
        idx = self.mid_air * ~self.has_jumped * self.was_in_flight
        # idx = torch.logical_and(self.mid_air,~self.has_jumped)
        self.max_height[idx] = torch.max(self.max_height[idx],self.root_states[idx, 2]) # update max height achieved
        
        self.min_height[~self.has_jumped] = torch.min(self.min_height[~self.has_jumped], self.root_states[~self.has_jumped, 2]) # update min height achieved

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()
        if self.viewer and self.enable_viewer_sync and self.cfg.env.debug_draw:
            self._draw_debug_goal()

        

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.linalg.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)

        # Reset if the robot base is too close to the ground:
        self.reset_buf[self.root_states[:,2] - self.get_terrain_height(self.root_states[:,:2]).flatten() <= self.cfg.env.reset_height] = True # check if the robot base is below 0.1 metres
    
        if self.additional_termination_conditions:
            self.reset_buf[self.ori_error > self.cfg.env.reset_orientation_error] = True # Reset if the orientation error is too big
            self.reset_buf[self.reset_idx_landing_error] = True # Reset agent if landing error is big

            # Reset if agent moves too far after landing:
            # OR have been initialised as jumped and moved too much from INITIAL position
            idx = self.settled_after_init * self.has_jumped * self._has_jumped_rand_envs
            if self.jump_type == "forward_with_obstacles":
                post_landing_error = 0.0*torch.linalg.norm(self.root_states[:, :2] - self.landing_poses[:, :2], dim=-1)
            else:
                post_landing_error = torch.linalg.norm(self.root_states[:, :2] - self.landing_poses[:, :2], dim=-1)
            # post_landing_error = torch.zeros_like(self.root_states[:,0])
            post_landing_error[idx] = torch.linalg.norm(self.root_states[idx,:3] - self.initial_root_states_nonrandomised[idx,:3],dim=1)
            self.reset_buf[torch.logical_and(self.has_jumped,post_landing_error>0.1)] = True

            idx = torch.logical_and(self.has_jumped,~self._has_jumped_rand_envs)

            # self.reset_buf[idx * (self.max_height < 0.52)] = True

            # If any foot has moved moved more than 10cm post landing reset the episode:
            
            # idx = torch.logical_or(torch.logical_and(self.episode_length_buf - self.settled_after_init_timer < 10, self.settled_after_init),
                                    # torch.logical_and(self.episode_length_buf - self._has_jumped_switched_time < 10, self.settled_after_init))

            # idx = self.has_jumped

            # feet_pos_ini_global = self.initial_root_states_nonrandomised[:,:3].view(-1,1,3) + torch.tensor(self.cfg.init_state.rel_foot_pos).to(self.device).transpose(1,0).view(1,4,3)
            err = torch.norm(self.feet_pos - self.landing_foot_poses,dim=-1)
            # self.reset_buf[torch.logical_and(idx,torch.any(err>0.1,dim=1))] = True
            # self.reset_buf[torch.logical_and(idx,torch.any(~self.contacts,dim=1))] = True
            
            # Reset if downwards acceleration for body is too large:
            # idx = (self.episode_length_buf - self.settled_after_init_timer < 20) * \
            # (self.episode_length_buf>3) * self.settled_after_init
            
            # self.reset_buf[torch.logical_and(idx,self.base_acc[:,2] < -8.)] = True
            self.reset_buf[torch.any(torch.abs((self.actions - self.last_actions)/self.dt) > 600,dim=-1)] = True
            # self.reset_buf[torch.any(torch.abs((self.actions - 2*self.last_actions + self.last_last_actions) / (self.dt**2)) > 30000,dim=-1)] = True

            grav_acc_violated = (self.base_acc[:,2] < -9.81) * (self.base_acc_prev[:,2] < -9.81) * ~torch.all(~self.contacts,dim=-1)#~self.mid_air
            self.reset_buf[grav_acc_violated] = True        

        # If joint velocity exceeds a threshold, reset:
        # self.reset_buf[torch.logical_and(~self.was_in_flight,torch.any((torch.abs(self.dof_vel) - self.dof_vel_limits) > 10,dim=-1))] = True

        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def check_jump(self):
        """ Check if the robot has jumped
        """
        
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        # contact_filt = contact
        contact_filt = torch.logical_or(contact, self.last_contacts) # Contact is true only if either current or previous contact was true
        self.contact_filt = contact_filt.clone() # Store it for the rewards that use it

        # Handle starting in mid-air (initialise in air):
        settled_after_init = torch.logical_and(torch.all(contact_filt,dim=1), self.root_states[:,2]<=0.4)
        jump_filter = torch.all(~contact_filt, dim=1)#torch.logical_and(torch.all(~contact_filt, dim=1),self.root_states[:,2]>0.32) # If no contact for all 4 feet, jump is true
        # jump_filter = torch.sum(contact_filt, dim=1) <= 2 # If more than  foot is in the air, jump has started

        self.mid_air = jump_filter.clone()

        idx_record_pose = torch.logical_and(settled_after_init,~self.settled_after_init)
        self.initial_foot_poses[idx_record_pose] = self.feet_pos[idx_record_pose].clone()
        # Record the time at which the robot settled after initialisation:
        self.settled_after_init_timer[idx_record_pose] = self.episode_length_buf[idx_record_pose].clone()

        self.settled_after_init[settled_after_init] = True


        # Only consider in flight if robot has settled after initialisation and is in the air:
        # (only switched to true once for each robot per episode)
        self.was_in_flight[torch.logical_and(jump_filter,self.settled_after_init)] = True # If no contact for all 4 feet, robot is in flight


        # The robot has already jumped IFF it was previously in flight and has now landed:
        has_jumped = torch.logical_and(torch.any(contact_filt,dim=1), self.was_in_flight) 
       
        # Record landing pose after first jump (before self.has_jumped is updated):
        self.landing_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.root_states[torch.logical_and(has_jumped,~self.has_jumped),:7]
        self.landing_foot_poses[torch.logical_and(has_jumped,~self.has_jumped)] = self.feet_pos[torch.logical_and(has_jumped,~self.has_jumped),:,:]

        # Only count the first time flight is achieved:
        self.has_jumped[has_jumped] = True 

        env_ids = self.not_pushed * ~self.has_jumped * self.was_in_flight
        # env_ids = torch.logical_and(env_ids, self.root_states[:,2] > 0.6)
        
        if self.cfg.domain_rand.push_towards_goal and torch.any(env_ids):
            self._push_robots_desired(env_ids)
            self.not_pushed[env_ids] = False        

        env_ids = torch.logical_and(self.push_upwards_envs,self.episode_length_buf == self.push_upwards_timer)
        if self.cfg.domain_rand.push_upwards and torch.any(env_ids):
            self._push_robots_upwards(env_ids)
            self.not_pushed[env_ids] = False   


    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """

        if len(env_ids) == 0:
            return
        # Store whether the robot was successful or was terminated last episode:
        if self.init_done:
            self.success_rate[env_ids] = torch.roll(self.success_rate[env_ids], 1, dims=-1)
            # If reset set to 0, if successful set to 1 (-1 by default)
            self.success_rate[env_ids,0] = self.has_jumped[env_ids] * (~(self.reset_buf[env_ids].bool()*~self.time_out_buf[env_ids].bool())).float()
        # self.terrain_levels[env_ids] = 5
        # self.terrain_types[env_ids] = 0
        # self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        # self.env_properties[env_ids] = self.terrain_properties[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        
        # update curriculum
        if self.jump_type == "forward_with_obstacles" and self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        
        if self.cfg.env.pass_remaining_time:
            self.remaining_time[env_ids] = self.max_episode_length

        if self.cfg.control.use_action_filter:
            self._reset_action_filter(env_ids)

        # Check if robots should be reset for continuous jumping:
        if self.cfg.env.continuous_jumping:
            reset_proba = self.cfg.env.continuous_jumping_reset_probability
            reset_prob_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor([reset_proba],device=self.device))
            self.continuous_jump_reset_prob[env_ids] = reset_prob_distr.sample((len(env_ids),1)).flatten()

            reset_env_ids = torch.logical_or(self.reset_buf* ~self.time_out_buf, self.reset_buf * self.continuous_jump_reset_prob)
            self.cont_jump_reset_env_ids = torch.nonzero(reset_env_ids,as_tuple=False).flatten()

            
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        # Reset stored states for those environments
        self._reset_stored_states(env_ids)

        # At each reset, reset the spring parameters (or leave them as default if no randomisation):
        self._reset_spring_params(env_ids)


        # Simulate episodic latency:
        if self.cfg.domain_rand.sim_latency:
            # self.episodic_latency[env_ids] = 1e-3 * self.cfg.domain_rand.base_latency
            self.episodic_latency[env_ids] = 1e-3*torch_rand_float(self.latency_range[0],self.latency_range[1], (len(env_ids), 1), device=self.device).flatten()
        if self.cfg.domain_rand.sim_pd_latency:
            self.episodic_pd_latency[env_ids] = 1e-3*torch_rand_float(self.pd_latency_range[0],self.pd_latency_range[1], (len(env_ids), 1), device=self.device).flatten()


        # Randomise the rigid body parameters (ground friction, restitution, etc.):
        self.randomize_rigid_body_props(env_ids)
        # This randomises them at every reset (slows down sim a lot so rn it's only randomising
        # at env creation)
        # self._refresh_actor_rigid_shape_props(env_ids)

        # Randomize joint parameters:
        self.randomize_dof_props(env_ids)

        self._refresh_actor_dof_props(env_ids)


        # Recompute commands
        self.commands[env_ids,:],self.command_vels[env_ids,:] = self._recompute_commands(env_ids)

        self.was_in_flight[env_ids] = False
        self.mid_air[env_ids] = False
        self.has_jumped[env_ids] = False
        self.settled_after_init[env_ids] = False
        self.landing_poses[env_ids,:] = float('nan')#1e4 + self.root_states[env_ids,:7].clone()
        self.landing_foot_poses[env_ids] = self.feet_pos[env_ids,:,:].clone()
        self.not_pushed[env_ids] = True
        self.reset_idx_landing_error[env_ids] = False
        self._has_jumped_rand_envs[env_ids] = False


        # self._resample_commands(env_ids)

        if self.cfg.domain_rand.randomize_has_jumped:
            # Only affects those that didnt start with randomised pos/vel:
            env_ids_not_rand = env_ids[self._pos_vel_rand_envs[env_ids] == 0]
            self.has_jumped[env_ids_not_rand] = self.has_jumped_randomisation_prob.sample((len(env_ids_not_rand),1)).bool().flatten()
            self._has_jumped_rand_envs[env_ids_not_rand] = self.has_jumped[env_ids_not_rand] == True
            # Idx of environments that have has_jumped as true now:
            idx = env_ids_not_rand[self._has_jumped_rand_envs[env_ids_not_rand] == 1]
            self._reset_randomised_has_jumped_timer[idx] = torch_rand_float(1.0,0.3*self.max_episode_length,(len(idx),1),device=self.device).int().flatten()
            # If allowing them to jump in the final part of the episode - just don't allow them as there isnt enough time.
            # max_episode_length = self.max_episode_length.clip(max=3/self.dt)
            # self._reset_randomised_has_jumped_timer[self._reset_randomised_has_jumped_timer>0.4*max_episode_length] = 0.4*max_episode_length
            self._has_jumped_switched_time[idx] = self.max_episode_length
            # self.landing_poses[idx] = self.root_states[idx,:7].clone()

        self.push_upwards_envs[env_ids] = self.push_upwards_distr.sample((len(env_ids),1)).bool().flatten()
        self.push_upwards_timer[env_ids] = torch_rand_float(1.0,0.1*self.max_episode_length,(len(env_ids),1),device=self.device).int().flatten()

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_dof_acc[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.last_feet_vel[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.max_height[env_ids] = self.base_init_state[2]#self.root_states[env_ids, 2]
        self.min_height[env_ids] = self.base_init_state[2]#self.root_states[env_ids, 2]
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1

        # Reset state history for those environments
        if self.cfg.env.use_state_history:
            self._reset_state_history(env_ids)
        
        # Reset episode storage of some data:
        # self.dof_acc_stored[env_ids] = 0.
        # self.base_acc_stored[env_ids] = 0.
        # self.action_rate_stored[env_ids] = 0.
        # self.action_rate_2_stored[env_ids] = 0.
        # fill extras
        self.extras["episode"] = {}
        
        for key in self.episode_sums.keys():

            # idx = env_ids[~(self.reset_buf[env_ids]* ~self.time_out_buf[env_ids])]
            idx = env_ids[self.time_out_buf[env_ids]]

            if idx is None or len(idx) == 0:
                val = 0.
            else:
                val = torch.mean(self.episode_sums[key][idx])
            if key[:4] == "task":
                self.extras["episode"]['rew_' + key] =  val / self.max_episode_length_s
            else:
                self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        # if self.cfg.terrain.curriculum:
            # self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        # if self.cfg.commands.curriculum:
        #     self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        for i in range(len(self.lag_buffer)):
            self.lag_buffer[i][env_ids, :] = 0
    
    def _reset_action_filter(self, env_ids):
        # if self.cfg.control.filter_type == "EMA":
        #     if self.cfg.control.control_type == "P_joint_pos":
        #         self.action_filter[env_ids,:] = self.default_dof_pos
        #     else:
        #         self.action_filter[env_ids,:] = 0.0
        # elif self.cfg.control.filter_type == "butterworth":
        self.action_filter.reset(env_ids,control_type=self.cfg.control.control_type)


    def _filter_actions(self,actions):
        # if self.cfg.control.filter_type == "EMA":
        #     alpha = self.cfg.control.action_filter_alpha
        #     filtered_actions = (1-alpha)*self.action_filter + alpha*actions
        #     self.action_filter = filtered_actions.clone()

        # elif self.cfg.control.filter_type == "butterworth":
        filtered_actions = self.action_filter.filter(actions)


        return filtered_actions
    def _store_PD_states(self):
        '''
        Store the dof pos and vels at a higher frequency (for PD control ONLY).
        '''
        self.pd_dof_pos_stored = torch.roll(self.pd_dof_pos_stored, 1, dims=2)
        self.pd_dof_pos_stored[:,:,0] = self.dof_pos.clone()
        self.pd_dof_vel_stored = torch.roll(self.pd_dof_vel_stored, 1, dims=2)
        self.pd_dof_vel_stored[:,:,0] = self.dof_vel.clone()

    def _store_states(self):
        '''
        Store the current states in the state storage buffers
        '''

        
        self.base_lin_vel_stored = torch.roll(self.base_lin_vel_stored, 1, dims=2)
        self.base_lin_vel_stored[:,:,0] = self.base_lin_vel_history.clone() 

        self.base_ang_vel_stored = torch.roll(self.base_ang_vel_stored, 1, dims=2)
        self.base_ang_vel_stored[:,:,0] = self.base_ang_vel_history.clone()

        self.root_states_stored = torch.roll(self.root_states_stored, 1, dims=2)
        self.root_states_stored[:,:,0] = self.root_states_history.clone()

        self.dof_pos_stored = torch.roll(self.dof_pos_stored, 1, dims=2)
        self.dof_pos_stored[:,:,0] = self.dof_pos_history.clone() 

        self.dof_vel_stored = torch.roll(self.dof_vel_stored, 1, dims=2)
        self.dof_vel_stored[:,:,0] = self.dof_vel_history.clone() 

        self.actions_stored = torch.roll(self.actions_stored, 1, dims=2)
        self.actions_stored[:,:,0] = self.actions_history.clone()

        self.contacts_stored = torch.roll(self.contacts_stored, 1, dims=2)
        self.contacts_stored[:,:,0] = self.contacts_history.clone()

        self.base_quat_stored = torch.roll(self.base_quat_stored, 1, dims=2)
        self.base_quat_stored[:,:,0] = self.base_quat_history.clone() 

        self.ori_error_stored = torch.roll(self.ori_error_stored, 1, dims=2)
        self.ori_error_stored[:,:,0] = self.ori_error_history.clone()

        # self.error_quat_stored = torch.roll(self.error_quat_stored, 1, dims=2)
        # self.error_quat_stored[:,:,0] = self.error_quat_history.clone()
        
        # self.force_sensor_stored = torch.roll(self.force_sensor_stored, 1, dims=3)
        # self.force_sensor_stored[:,:,:,0] = self.force_sensor_readings_history.clone()
        
        self.has_jumped_stored = torch.roll(self.has_jumped_stored, 1, dims=2)
        self.has_jumped_stored[:,:,0] = self.has_jumped_history.clone()

    def _compute_state_history(self):
        '''
        Compute the state history for the current timestep. This is done by shifting the current 
        history by one step and adding the current (possibly delayed) state to the first position.
        '''

        add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
           
        
         
        base_lin_vel = self.base_lin_vel.clone() + \
         add_noise * noise_scales.lin_vel * noise_level * (2 * torch.rand_like(self.base_lin_vel) - 1) 
        
        base_ang_vel = self.base_ang_vel.clone() + \
            add_noise * noise_scales.ang_vel * noise_level * (2 * torch.rand_like(self.base_ang_vel) - 1)
        
        root_states = self.root_states.clone()

        dof_pos = self.dof_pos.clone() + \
            add_noise * noise_scales.dof_pos * noise_level * (2 * torch.rand_like(self.dof_pos) - 1)
        
        dof_vel = self.dof_vel.clone() + \
            add_noise * noise_scales.dof_vel * noise_level * (2 * torch.rand_like(self.dof_vel) - 1)
        
        actions = self.actions.clone()

        noise_prob = self.cfg.noise.noise_scales.contacts_noise_prob
        noise_prob_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor([noise_prob],device=self.device))
        # If contact = 1 then 10% of the time it will be 0.
        # If contact = 0 then nothing happens
        contacts = self.contacts.clone() * \
            (1 - noise_prob_distr.sample((self.num_envs,4)).reshape(self.num_envs,-1))

        base_quat = self.base_quat.clone() + \
            add_noise * noise_scales.quat * noise_level * (2 * torch.rand_like(self.base_quat) - 1)
        
        ori_error = self.ori_error.clone().unsqueeze(-1) + \
            add_noise * noise_scales.ori_error * noise_level * (2 * torch.rand_like(self.ori_error.unsqueeze(-1)) - 1)
        
        # error_quat = self.error_quat.clone() + \
        #     add_noise * noise_scales.error_quat * noise_level * (2 * torch.rand_like(self.error_quat) - 1)
        
        has_jumped = self.has_jumped.clone().unsqueeze(-1)


        self.base_lin_vel_history = torch.roll(self.base_lin_vel_history, self.base_lin_vel.shape[-1], dims=1)
        self.base_lin_vel_history[:,0:self.base_lin_vel.shape[-1]] = base_lin_vel 
        
        self.base_ang_vel_history = torch.roll(self.base_ang_vel_history, self.base_ang_vel.shape[-1], dims=1)
        self.base_ang_vel_history[:,0:self.base_ang_vel.shape[-1]] = base_ang_vel
        
        self.root_states_history = torch.roll(self.root_states_history, 3, dims=1)
        self.root_states_history[:,0:3] = root_states[:,:3]
        
        self.dof_pos_history = torch.roll(self.dof_pos_history, self.dof_pos.shape[-1], dims=1)
        self.dof_pos_history[:,0:self.dof_pos.shape[-1]] = dof_pos 
        
        self.dof_vel_history = torch.roll(self.dof_vel_history, self.dof_vel.shape[-1], dims=1)
        self.dof_vel_history[:,0:self.dof_vel.shape[-1]] = dof_vel 

        self.actions_history = torch.roll(self.actions_history, self.actions.shape[-1], dims=1)
        self.actions_history[:,0:self.actions.shape[-1]] = actions
        
        self.contacts_history = torch.roll(self.contacts_history, self.contacts.shape[-1], dims=1)
        self.contacts_history[:,0:self.contacts.shape[-1]] = contacts
        
        self.base_quat_history = torch.roll(self.base_quat_history, self.base_quat.shape[-1], dims=1)
        self.base_quat_history[:,0:4] = base_quat 
        
        self.ori_error_history = torch.roll(self.ori_error_history, self.ori_error.shape[-1], dims=1)
        self.ori_error_history[:,0] = ori_error.squeeze(-1)
        
        # self.error_quat_history = torch.roll(self.error_quat_history, self.error_quat.shape[-1], dims=1)
        # self.error_quat_history[:,0:4] = error_quat 

        self.has_jumped_history = torch.roll(self.has_jumped_history, self.has_jumped.shape[-1], dims=1)
        self.has_jumped_history[:,0] = has_jumped.squeeze(-1)

    def _compute_yaw_error(self,q1,q2):
        """ Compute the yaw error between two quaternions
        """
        _,_,yaw1 = get_euler_xyz(q1)
        # print(yaw1[0])
        _,_,yaw2 = get_euler_xyz(q2)
        yaw_error = (yaw2 - yaw1) % (2*torch.pi)
        # print(yaw_error[0])
        yaw_error[yaw_error<-torch.pi] += 2*np.pi
        yaw_error[yaw_error>torch.pi] -= 2*np.pi

        return yaw_error

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        self.rew_buf_pos[:] = 0.
        self.rew_buf_neg[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew

            if torch.sum(rew) >= 0:
                self.rew_buf_pos += rew
            elif torch.sum(rew) <= 0:
                self.rew_buf_neg += rew
            self.episode_sums[name] += rew

            self.reward_logs[name].append(torch.mean(rew)) 
        
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)

        elif self.cfg.rewards.only_positive_rewards_ji22_style:
            self.rew_buf[:] = self.rew_buf_pos[:] * torch.exp(self.rew_buf_neg[:] / self.sigma_rew_neg)
        
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
    
    def compute_observations(self):
        """ Computes observations
        """

        hist_len = self.cfg.env.state_history_length
        self.obs_buf = torch.cat((  self.base_lin_vel_delayed * self.obs_scales.lin_vel,
                        self.base_ang_vel_delayed  * self.obs_scales.ang_vel,
                        # self.projected_gravity,
                        (self.dof_pos_delayed - self.default_dof_pos.repeat(1,hist_len)) * self.obs_scales.dof_pos,
                        self.dof_vel_delayed * self.obs_scales.dof_vel,
                        self.actions_delayed
                        ),dim=-1)

        if self.cfg.env.known_quaternion:
            self.obs_buf = torch.cat((self.obs_buf, self.base_quat_delayed*self.obs_scales.quat),dim=-1)
        if self.cfg.env.known_ori_error:
            self.obs_buf = torch.cat((self.obs_buf, self.ori_error_delayed*self.obs_scales.ori_error),dim=-1)
        # if self.cfg.env.known_error_quaternion:
        #     self.obs_buf = torch.cat((self.obs_buf, self.error_quat_delayed*self.obs_scales.error_quat),dim=-1)
        if self.cfg.env.jumping_target:
            self.obs_buf = torch.cat((self.obs_buf, self.commands[:, :]), dim=-1) # Relative distance to desired landing point
        if self.cfg.env.known_height:
            self.obs_buf = torch.cat((self.obs_buf, self.root_states_delayed[:, 2::3] * self.obs_scales.height), dim=-1) # Get the height of the robot
        if self.cfg.env.pass_remaining_time:
            self.obs_buf = torch.cat((self.obs_buf, self.remaining_time.unsqueeze(1)), dim=-1)
        if self.cfg.env.pass_has_jumped:
            self.obs_buf = torch.cat((self.obs_buf, self.has_jumped_delayed.unsqueeze(1)), dim=-1)

        if self.cfg.env.known_contact_feet:
            # Get the starting index of the contacts (to adapt its noise)
            contacts_idx = self.obs_buf.shape[-1]

            self.obs_buf = torch.cat((self.obs_buf, self.contacts_delayed), dim=-1)

        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat((self.obs_buf, heights), dim=-1)
        # add noise if needed
        # if self.add_noise:
        #     if self.cfg.noise.distribution == "uniform":
        #         noise = (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec_no_history
        #         self.obs_buf += noise
        #         self.noise_stored = noise.clone()
        #         if self.cfg.env.known_contact_feet:
        #             noise_prob = self.cfg.noise.noise_scales.contacts_noise_prob
        #             noise_prob_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor([noise_prob],device=self.device))
        #             self.obs_buf[:,contacts_idx:contacts_idx+4] = 1 - noise_prob_distr.sample((self.num_envs,4)).reshape(self.num_envs,-1)
        #     elif self.cfg.noise.distribution == "gaussian":
        #         # Get the standard deviation of the uniform distribution and use it as the std of the gaussian
        #         gaussian_std = torch.sqrt((2*self.noise_scale_vec)**2/12)
        #         self.obs_buf += torch.normal(torch.zeros_like(self.noise_scale_vec_no_history),gaussian_std).flatten()

    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        for s in range(len(props)):
            props[s].friction = self.friction_coeffs[env_id, 0]
            props[s].restitution = self.restitutions[env_id, 0]

        return props

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_limits_urdf = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_pos_limits_urdf[i] = self.dof_pos_limits[i].clone()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit


        return props

    def randomize_dof_props(self, env_ids):

        # Randomise the motor strength:
        if self.cfg.domain_rand.randomize_motor_strength:
            motor_strength_ranges = self.cfg.domain_rand.ranges.motor_strength_ranges
            self.motor_strengths[env_ids] = 1 * torch_rand_float(motor_strength_ranges[0], motor_strength_ranges[1], (len(env_ids),12), device=self.device)

        if self.cfg.domain_rand.randomize_motor_offset:
            min_offset, max_offset = self.cfg.domain_rand.ranges.motor_offset_range
            self.motor_offsets[env_ids, :] = torch_rand_float(min_offset, max_offset, (len(env_ids),12), device=self.device)

        if self.cfg.domain_rand.randomize_PD_gains:
            p_gains_range = self.cfg.domain_rand.ranges.p_gains_range
            d_gains_range = self.cfg.domain_rand.ranges.d_gains_range

            self.p_gains[env_ids] = self.cfg.control.stiffness["joint"] * torch_rand_float(p_gains_range[0], p_gains_range[1], (len(env_ids),12), device=self.device)
            self.d_gains[env_ids] = self.cfg.control.damping["joint"] * torch_rand_float(d_gains_range[0], d_gains_range[1], (len(env_ids),12), device=self.device)
               
        if self.cfg.domain_rand.randomize_joint_friction:
            joint_friction_range = self.cfg.domain_rand.ranges.joint_friction_range
            self.joint_friction_coeffs[env_ids] = torch_rand_float(joint_friction_range[0], joint_friction_range[1], (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_joint_damping:
            joint_damping_range = self.cfg.domain_rand.ranges.joint_damping_range
            self.joint_damping_coeffs[env_ids] = torch_rand_float(joint_damping_range[0], joint_damping_range[1], (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_joint_armature:
            joint_armature_range = self.cfg.domain_rand.ranges.joint_armature_range
            self.joint_armatures[env_ids] = torch_rand_float(joint_armature_range[0], joint_armature_range[1], (len(env_ids), 1), device=self.device)


    def _process_rigid_body_props(self, props, env_id):
        # props[0].mass += 1 # Add mass of the box
        # Total mass:
        self.total_mass = sum([prop.mass for prop in props])
        self.default_body_mass = props[0].mass

        if self.cfg.domain_rand.randomize_base_mass:
            props[0].mass = self.default_body_mass + self.payload_masses[env_id]

        if self.cfg.domain_rand.randomize_link_mass:
            for i in range(1, len(props)):
                props[i].mass *= self.link_masses[env_id, i-1]

        if self.cfg.domain_rand.randomize_com:
            # From Walk These Ways:
            props[0].com = gymapi.Vec3(self.com_displacements[env_id, 0], self.com_displacements[env_id, 1],
                                    self.com_displacements[env_id, 2])
        
            
        return props
  

    def _refresh_actor_dof_props(self, env_ids):

        for env_id in env_ids:
            dof_props = self.gym.get_actor_dof_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                dof_props["friction"][i] = self.joint_friction_coeffs[env_id, 0]
                dof_props["damping"][i] = self.joint_damping_coeffs[env_id, 0]
                dof_props["armature"][i] = self.joint_armatures[env_id, 0]

            self.gym.set_actor_dof_properties(self.envs[env_id], 0, dof_props)

    def _refresh_actor_rigid_shape_props(self, env_ids):
        ''' Refresh the rigid shape properties of the actor in the given environments, i.e.
            set the friction and restitution coefficients to the desired values.
        '''
        # From Walk These Ways:
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(self.num_dof):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitutions[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)
            
           


    def randomize_rigid_body_props(self, env_ids):
        ''' Randomise some of the rigid body properties of the actor in the given environments, i.e.
            sample the mass, centre of mass position, friction and restitution.'''
        # From Walk These Ways:
        if self.cfg.domain_rand.randomize_base_mass:
            min_payload, max_payload = self.cfg.domain_rand.ranges.added_mass_range

            self.payload_masses[env_ids] = torch_rand_float(min_payload, max_payload, (len(env_ids), 1), device=self.device).flatten()
        
        if self.cfg.domain_rand.randomize_link_mass:
            min_link_mass, max_link_mass = self.cfg.domain_rand.ranges.added_link_mass_range

            self.link_masses[env_ids] = torch_rand_float(min_link_mass, max_link_mass, (len(env_ids), self.num_bodies-1), device=self.device)

        if self.cfg.domain_rand.randomize_com:
            min_com_displacement, max_com_displacement = self.cfg.domain_rand.ranges.com_displacement_range
            self.com_displacements[env_ids, :] = torch_rand_float(min_com_displacement, max_com_displacement, (len(env_ids), 3), device=self.device)

        if self.cfg.domain_rand.randomize_friction:
            min_friction, max_friction = self.cfg.domain_rand.ranges.friction_range
            self.friction_coeffs[env_ids, :] = torch_rand_float(min_friction, max_friction, (len(env_ids), 1), device=self.device)

        if self.cfg.domain_rand.randomize_restitution:
            min_restitution, max_restitution = self.cfg.domain_rand.ranges.restitution_range
            self.restitutions[env_ids] = torch_rand_float(min_restitution, max_restitution, (len(env_ids), 1), device=self.device)

    def _randomize_gravity(self, external_force = None):

        if external_force is not None:
            self.gravities[:, :] = external_force.unsqueeze(0)
        elif self.cfg.domain_rand.randomize_gravity:
            min_gravity, max_gravity = self.cfg.domain_rand.ranges.gravity_range
            external_force = torch.rand(3, dtype=torch.float, device=self.device,
                                        requires_grad=False) * (max_gravity - min_gravity) + min_gravity

            self.gravities[:, :] = external_force.unsqueeze(0)

        sim_params = self.gym.get_sim_params(self.sim)
        gravity = self.gravities[0, :] + torch.Tensor([0, 0, -9.81]).to(self.device)
        sim_params.gravity = gymapi.Vec3(gravity[0], gravity[1], gravity[2])
        self.gym.set_sim_params(self.sim, sim_params)

    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.common_step_counter % int(self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity()
        if int(self.common_step_counter - self.cfg.domain_rand.gravity_rand_duration) % int(
                self.cfg.domain_rand.gravity_rand_interval) == 0:
            self._randomize_gravity(torch.tensor([0, 0, 0]))

    def _recompute_commands(self,env_ids):
        """ Recompute relative distance for the jumps:

    #     """
        commands = torch.zeros_like(self.commands)
        command_vels = torch.zeros_like(self.command_vels)

        dx = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        dy = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        dz = torch.zeros((self.num_envs, 1), device=self.device).flatten()
        
        if self.jump_type == "forward_with_obstacles": # self.cfg.terrain.mesh_type == "trimesh":
            # Compute the relative distance to the start of the box
            box_widths = self.env_properties[env_ids,0]
            dx[env_ids] = (self.env_origins[env_ids,0] - self.initial_root_states[env_ids,0] - box_widths/2)
            dy[env_ids] = (self.env_origins[env_ids,1] - self.initial_root_states[env_ids,1])

            # THen add a random offset to the x and y components:
            dx[env_ids] += torch_rand_float(0.2,0.4, (len(env_ids), 1), device=self.device).flatten()
            dy[env_ids] += torch_rand_float(-0.1,0.1, (len(env_ids), 1), device=self.device).flatten()

            range_dx = [self.cfg.commands.ranges.pos_dx_ini[0], self.cfg.commands.ranges.pos_dx_ini[1]]
            forward_idx = env_ids * (self.env_properties[env_ids,2] == 0.0)
            
            dx[forward_idx] = torch_rand_float(range_dx[0],range_dx[1], (len(forward_idx),1), device=self.device).flatten()
            
            range_dy = [self.cfg.commands.ranges.pos_dy_ini[0], self.cfg.commands.ranges.pos_dy_ini[1]]
            dy[forward_idx] = torch_rand_float(range_dy[0],range_dy[1], (len(forward_idx),1), device=self.device).flatten()

        elif self.jump_type == "forward" and self.cfg.commands.curriculum:
            up_jump_envs = self.up_jump_distribution.sample((len(env_ids),1)).flatten()
            env_ids_up_jump = env_ids[up_jump_envs==1]

            range_dx = torch.stack((self.cfg.commands.ranges.pos_dx_ini[0] * (self.command_dist_levels - 1) / self.cfg.commands.num_levels,
            self.cfg.commands.ranges.pos_dx_ini[1] * self.command_dist_levels / self.cfg.commands.num_levels)).clip(min=0.0)

            # For dy halve the number of levels (since the range is generally smaller)
            range_dy_levels = (self.command_dist_levels / int(self.cfg.commands.num_levels/1.5)).clip(max=1.0)
            range_dy = torch.stack((self.cfg.commands.ranges.pos_dy_ini[0] * range_dy_levels,
            self.cfg.commands.ranges.pos_dy_ini[1] * range_dy_levels))
            # range_dy = self.cfg.commands.ranges.pos_dy_ini
            dx[env_ids] = torch_rand_float_tensor(range_dx[0,env_ids], range_dx[1,env_ids], (len(env_ids),), device=self.device)
            dy[env_ids] = torch_rand_float_tensor(range_dy[0,env_ids], range_dy[1,env_ids], (len(env_ids),), device=self.device)
            
            dx[env_ids_up_jump] = 0.0
            dy[env_ids_up_jump] = 0.0
            
        elif self.cfg.commands.randomize_commands:
            if self.jump_type != "upward":
                up_jump_envs = self.up_jump_distribution.sample((len(env_ids),1)).flatten()
                env_ids_up_jump = env_ids[up_jump_envs==1]
                # For now only change the pos components:
                dx[env_ids] = torch_rand_float(self.pos_command_variation[0,0], self.pos_command_variation[1,0], (len(env_ids), 1), device=self.device).flatten()
                dy[env_ids] = torch_rand_float(self.pos_command_variation[0,1], self.pos_command_variation[1,1], (len(env_ids), 1), device=self.device).flatten()
                # dz = torch_rand_float(self.pos_command_variation[0,2], self.pos_command_variation[1,2], (len(env_ids), 1), device=self.device)
                dx[env_ids_up_jump] = 0.0
                dy[env_ids_up_jump] = 0.0

        else:
            dx[env_ids] = torch.zeros((len(env_ids), 1), device=self.device).flatten()
            dy[env_ids] = torch.zeros((len(env_ids), 1), device=self.device).flatten()
            # dz = torch.zeros((len(env_ids), 1), device=self.device)

        if self.jump_type == "forward_with_obstacles" and self.cfg.commands.jump_over_box:
            # When jumping over box, the robot should land 20cm behind the box
            # and box should be relatively thin (width<0.3)
            box_widths = self.env_properties[env_ids,0]
            env_ids_filtered = env_ids[box_widths <= 0.2]

            dx[env_ids_filtered] = (self.env_origins[env_ids_filtered,0] - self.initial_root_states[env_ids_filtered,0])
            dx[env_ids_filtered] += self.env_properties[env_ids_filtered,0]/2 + 0.25

        # dz depends on the height of the object that the robot is jumping on:
        if  self.jump_type == "forward_with_obstacles": #self.cfg.terrain.mesh_type == "trimesh":
            global_pose = torch.stack((dx[env_ids],dy[env_ids]),dim=1) + self.root_states[env_ids,:2]
            dz[env_ids] += self.get_terrain_height(global_pose).flatten()

        else:
            dz[env_ids] += self.env_origins[env_ids,2]

        commands[env_ids, 0] = dx[env_ids].squeeze() + self.command_distances["x"]
        commands[env_ids, 1] = dy[env_ids].squeeze() + self.command_distances["y"]
        commands[env_ids, 2] = dz[env_ids].squeeze() + self.command_distances["z"]


        des_angles_euler = torch.zeros((self.num_envs,3),device=self.device)

        # Compute information about the obstacle:
        if  self.jump_type == "forward_with_obstacles" and self.cfg.env.object_information:
            # Compute the relative distance to the centre of the box
            commands[env_ids, 7:10] = self.env_origins[env_ids,:] - self.initial_root_states[env_ids,:3]
            commands[env_ids, 7:10] += torch_rand_float(-0.05, 0.05, (len(env_ids), 3), device=self.device)
            # And add object properties (length,height,width)
            commands[env_ids, 10::] = self.env_properties[env_ids,:]
            commands[env_ids, 10::] += torch_rand_float(-0.05, 0.05, (len(env_ids), 3), device=self.device)
            # Change desired pitch based on the slope:       
            # des_angles_euler[env_ids,1] = wrap_to_pi(torch.atan2(self.env_properties[env_ids,0],self.env_properties[env_ids,2]))
            
        else:
            commands[env_ids, 7::] = 0.

        # Convert to quaternion:
        # des_angles_euler = torch.tensor(self.command_distances["des_angles_euler"]).view(3,1)
        # Desired yaw depends on the heading between starting point and goal
        initial_yaw = wrap_to_pi(get_euler_xyz(self.root_states[:,3:7])[2])
        
        des_angles_euler[:,2] = wrap_to_pi(torch.atan2(commands[:,1],commands[:,0]) - initial_yaw)
        if self.cfg.commands.randomize_yaw:
            des_angles_euler[:,2] += torch_rand_float(-np.pi/2, np.pi/2, (self.num_envs, 1), device=self.device).flatten()
            des_angles_euler[:,2] = wrap_to_pi(des_angles_euler[:,2])
            # des_angles_euler[:,2] = torch.clip(des_angles_euler[:,2], -np.pi/2, np.pi/2)
        if self.cfg.commands.distances.des_yaw is not None:
            des_angles_euler[:,2] = self.cfg.commands.distances.des_yaw

        self.des_angles_euler[env_ids] = des_angles_euler[env_ids]
        desired_quat = quat_from_euler_xyz(des_angles_euler[:,0],des_angles_euler[:,1],des_angles_euler[:,2])#.squeeze()
        
        commands[env_ids, 3] =  desired_quat[env_ids,0]
        commands[env_ids, 4] =  desired_quat[env_ids,1]
        commands[env_ids, 5] =  desired_quat[env_ids,2]
        commands[env_ids, 6] =  desired_quat[env_ids,3]

        # Update the desired velocities:

        # These have been derived based on best fit line on joint friction vs flight time from the upwards jump:
        a = -4.4207
        b = 0.5563
        flight_time = self.joint_friction_coeffs[env_ids] * a + b
        command_vels[env_ids,0:3] = commands[env_ids,0:3]/(flight_time)
        command_vels[env_ids,3:6] = des_angles_euler[env_ids,:]/(flight_time)

        return commands[env_ids],command_vels[env_ids]
 
    def inverse_kinematics_tensor(self,foot_pos,env_ids):
        """Get joint angles for leg legID with desired xyz position in leg frame.

        Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;

        From SpotMicro:
        https://github.com/OpenQuadruped/spot_mini_mini/blob/spot/spotmicro/Kinematics/LegKinematics.py
        """
        # rename links
        HIP_LINK_LENGTH = 0.0847
        THIGH_LINK_LENGTH = 0.213
        CALF_LINK_LENGTH = 0.213
        shoulder_length = HIP_LINK_LENGTH
        elbow_length = THIGH_LINK_LENGTH
        wrist_length = CALF_LINK_LENGTH

        # Need to reorder the feet/joint angles to match different notation:

        foot_pos = foot_pos[:,torch.tensor([1,0,3,2]),:]
        # coords
        x = foot_pos[:,:,0] - torch.tensor([0.1881,0.1881,-0.1881,-0.1881]).to(self.device)
        y = foot_pos[:,:,1] - torch.tensor([-0.04205,0.04205,-0.04205,0.04205]).to(self.device)
        z = foot_pos[:,:,2]



        # get_domain
        D = (y**2 + (-z) ** 2 - shoulder_length**2 + (-x) ** 2 - elbow_length**2 - wrist_length**2) / (
            2 * wrist_length * elbow_length
        )

        D = torch.clip(D, min=-1.0, max=1.0)

        # check Right vs Left leg for hip angle
        sideSign = 1
        sideSign = torch.ones((len(env_ids),4),device=self.device)
        sideSign[:,[0,2]] = -1

        # Right Leg Inverse Kinematics Solver
        wrist_angle = torch.atan2(-torch.sqrt(1 - D**2), D)
        sqrt_component = y**2 + (-z) ** 2 - shoulder_length**2

        sqrt_component[sqrt_component < 0.0] = 0.0

        shoulder_angle = -torch.atan2(z, y) - torch.atan2(torch.sqrt(sqrt_component), sideSign * shoulder_length)
        elbow_angle = torch.atan2(-x, torch.sqrt(sqrt_component)) - torch.atan2(
            wrist_length * torch.sin(wrist_angle), elbow_length + wrist_length * torch.cos(wrist_angle)
        )
        joint_angles = torch.stack((-shoulder_angle, elbow_angle, wrist_angle),dim=-1).view(-1,12)

        joint_angles_ordered = joint_angles[:, torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])]


        return joint_angles_ordered
    
    def _compute_jacobian_and_position(self, q, legID):
        """Get Jacobian and foot position of leg legID.
        Leg 0: FR; Leg 1: FL; Leg 2: RR ; Leg 3: RL;
        """

        q = q[legID * 3 : legID * 3 + 3]
        HIP_LINK_LENGTH = 0.0847
        THIGH_LINK_LENGTH = 0.213
        CALF_LINK_LENGTH = 0.213
        # rename links
        l1 = HIP_LINK_LENGTH
        l2 = THIGH_LINK_LENGTH
        l3 = CALF_LINK_LENGTH

        sideSign = 1
        if legID == 0 or legID == 2:
            sideSign = -1

        s1 = np.sin(q[0])
        s2 = np.sin(q[1])
        s3 = np.sin(q[2])

        c1 = np.cos(q[0])
        c2 = np.cos(q[1])
        c3 = np.cos(q[2])

        c23 = c2 * c3 - s2 * s3
        s23 = s2 * c3 + c2 * s3

        # J = np.zeros((3, 3))

        # J[1, 0] = -sideSign * l1 * s1 + l2 * c2 * c1 + l3 * c23 * c1
        # J[2, 0] = sideSign * l1 * c1 + l2 * c2 * s1 + l3 * c23 * s1
        # J[0, 1] = -l3 * c23 - l2 * c2
        # J[1, 1] = -l2 * s2 * s1 - l3 * s23 * s1
        # J[2, 1] = l2 * s2 * c1 + l3 * s23 * c1
        # J[0, 2] = -l3 * c23
        # J[1, 2] = -l3 * s23 * s1
        # J[2, 2] = l3 * s23 * c1

        # foot pos
        pos = np.zeros(3)
        pos[0] = -l3 * s23 - l2 * s2
        pos[1] = l1 * sideSign * c1 + l3 * (s1 * c23) + l2 * c2 * s1
        pos[2] = l1 * sideSign * s1 - l3 * (c1 * c23) - l2 * c1 * c2

        return pos

    def _sample_latency(self,pd=False):
        ''' Sample the latency for each environment based on the latency range in the cfg file
        
        Returns:
            idx (torch.tensor): The index of the closest observation based on the latency.
        '''
        # Sample latency:
        if pd:
            # sampled_latency = 1e-3*torch_rand_float(self.pd_latency_range[0],self.pd_latency_range[1], (self.num_envs, 1), device=self.device).flatten()
            sampled_latency = self.episodic_pd_latency
        else:
            # sampled_latency = 1e-3*torch_rand_float(self.latency_range[0],self.latency_range[1], (self.num_envs, 1), device=self.device).flatten()

            sampled_latency = self.episodic_latency.clone()
            sampled_latency += 1e-3*torch_rand_float(self.cfg.domain_rand.ranges.additional_latency_range[0],
                                                                        self.cfg.domain_rand.ranges.additional_latency_range[1],
                                                                        (self.num_envs, 1), device=self.device).flatten()
        # Check the stored observations for two adjacent observations that current time - sampled_latency 
        # falls between:
        # t_i <= t_c - sampled_latency <= t_i+1
            # Need to ensure that the latency at step t+1 doesn't exceed the latency at step t + dt:
            sampled_latency = torch.clip(sampled_latency, min=0.0)
            sampled_latency = torch.where(sampled_latency < (self.prev_sampled_latency + self.dt), sampled_latency,
                                          (self.prev_sampled_latency + self.dt))

            
        idx = torch.floor(sampled_latency / self.dt).long()

        self.prev_sampled_latency = sampled_latency.clone()
        
        return idx, sampled_latency
 
    def _get_delayed_PD_states(self,idx,sampled_latency):
        ''' Get the delayed states based on the sampled latency and the index of nearest observation.
        For the low-level PD control latency (uses dof_pos stored at a higher rate.)
        '''
        dof_pos_delayed = self._interpolate_obs(idx,sampled_latency,self.pd_dof_pos_stored)
        dof_vel_delayed = self._interpolate_obs(idx,sampled_latency,self.pd_dof_vel_stored)

        return dof_pos_delayed, dof_vel_delayed

    def _get_delayed_states(self,idx,sampled_latency):
        ''' Get the delayed states based on the sampled latency and the index of nearest observation.

        '''

        self.root_states_delayed = self._interpolate_obs(idx,sampled_latency,self.root_states_stored)
        
        # For quaternion need to use SLERP:
        self.base_quat_delayed = self._interpolate_obs(idx,sampled_latency,self.base_quat_stored,is_quat=True)
        # self.base_quat_delayed = self.base_quat_stored[:,:,0]# self._interpolate_obs(idx,sampled_latency,self.base_quat_stored,is_quat=True)
        # self.base_lin_vel_delayed = quat_rotate_inverse(self.base_quat_delayed, self.root_states_delayed[:, 7:10])
        # self.base_ang_vel_delayed = quat_rotate_inverse(self.base_quat_delayed, self.root_states_delayed[:, 10:13])
        self.base_lin_vel_delayed = self._interpolate_obs(idx,sampled_latency,self.base_lin_vel_stored)
        self.base_ang_vel_delayed = self._interpolate_obs(idx,sampled_latency,self.base_ang_vel_stored)
        self.dof_pos_delayed = self._interpolate_obs(idx,sampled_latency,self.dof_pos_stored)
        self.dof_vel_delayed = self._interpolate_obs(idx,sampled_latency,self.dof_vel_stored)
        self.contacts_delayed = torch.round(self._interpolate_obs(idx,sampled_latency,self.contacts_stored.int())).bool()
        self.actions_delayed = self.actions_stored[:,:,0]#self._interpolate_obs(idx,sampled_latency,self.actions_stored)
        # self.ori_error_delayed =  quat_distance(self.base_quat_delayed.reshape(-1,4),
        #                                          self.commands[:, 3:7].repeat(1,self.cfg.env.state_history_length).reshape(-1,4),as_quat=False)
        # self.error_quat_delayed = quat_distance(self.base_quat_delayed.reshape(-1,4), 
        #                                         self.commands[:, 3:7].repeat(1,self.cfg.env.state_history_length).reshape(-1,4),as_quat=True)
        # self.force_sensor_delayed[:,:,2] = self._interpolate_obs(idx,sampled_latency,self.force_sensor_stored[:,:,2,:]) # Only vertical component
        # self.contacts_delayed = self.force_sensor_delayed[:,:,2] > 1
        # self.has_jumped_delayed = self._interpolate_obs(idx,sampled_latency,self.has_jumped_stored.int()).bool()
        self.has_jumped_delayed = self.has_jumped.clone()
    def _interpolate_obs(self,idx,sampled_latency,obs_history,is_quat=False):
        ''' Interpolate between two observations based on the latency and the index of the closest observation.
        For quaternions use SLERP.

        Args: 
            idx (torch.tensor): The index of the closest observation based on the latency.
            obs_history (torch.tensor): The history of observations.
            is_quat (bool): If the observation is a quaternion.

        Returns:
            obs_interp (torch.tensor): The interpolated observation.
        '''

        obs_history = obs_history.clone().transpose(2,1)

        obs_prev = obs_history[torch.arange(len(obs_history)),idx+1] # First interpolation observation
        obs_next = obs_history[torch.arange(len(obs_history)),idx] # Second interpolation observation

        if not is_quat:
            obs_interp = obs_next + (obs_prev-obs_next)*((sampled_latency - idx*self.dt)/self.dt).unsqueeze(1)
        else:
            # obs_next = obs_next.reshape(-1,4)
            # obs_prev = obs_prev.reshape(-1,4)
            # Use SLERP for interpolating quaternions:
            # steps = ((sampled_latency - idx*self.dt)/self.dt).repeat(1,self.cfg.env.state_history_length).reshape(-1,1)

            # obs_interp = quat_slerp(obs_next,obs_prev,steps).reshape(self.num_envs,-1)

            # Just pass the previous observation for now (SLERP causes a memory leak for some reason):
            obs_interp = obs_prev.clone()
            

        return obs_interp


    def _get_spring_params(self):
        """ Get spring parameters

        Returns:
            [torch.Tensor]: Spring stiffness
            [torch.Tensor]: Spring damping
            [torch.Tensor]: Spring rest position

            as size (num_envs,num_dof)
        """
        k = torch.zeros((self.num_envs,self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)
        b = torch.zeros((self.num_envs,self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)
        q0 = torch.zeros((self.num_envs,self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)
        
        k[:] = self.cfg.init_state.spring_stiffness
        b[:] = self.cfg.init_state.spring_damping
        q0[:] = self.cfg.init_state.spring_rest_pos

        return k, b, q0

    def _compute_spring_torques(self):
        """ Compute spring torques based on the current joint positions and velocities

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # spring torques
        spring_torques = torch.zeros((self.num_envs,self.num_dof), dtype=torch.float, device=self.device, requires_grad=False)
        # k, b, q0 = self._get_spring_params()
        k, b, q0 = self.spring_stiffness.clone(), self.spring_damping.clone(), self.spring_rest_pos.clone()


        q = self.dof_pos.clone()
        dq = self.dof_vel.clone()
        
        # For hips the two sides have reverse directions
        left_hip_inactive = q[:,0::6] <= q0[:,0::6]
        right_hip_inactive = q[:,3::6] >= q0[:,3::6]
    
        hip_inactive = torch.stack([left_hip_inactive[:,0],right_hip_inactive[:,0],left_hip_inactive[:,1],right_hip_inactive[:,1]],dim=1)

        # left_hip_inactive = q[:,3::6] <= q0[:,3::6]
        # right_hip_inactive = q[:,0::6] >= q0[:,0::6]
    
        # hip_inactive = torch.stack([right_hip_inactive[:,0],left_hip_inactive[:,0],right_hip_inactive[:,1],left_hip_inactive[:,1]],dim=1)


        thigh_inactive = q[:,1::3] <= q0[:,1::3]
        calf_inactive = q[:,2::3] >= q0[:,2::3]
        inactive_conditions = torch.stack([hip_inactive,thigh_inactive,calf_inactive],dim=1).transpose(1,2).reshape(-1,12)

        # Set stiffness and damping for inactive springs to 0
        k[inactive_conditions] = 0
        b[inactive_conditions] = 0
            
        spring_torques = -k * (q - q0) - b * dq


        return spring_torques
    

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        action_scale = torch.tensor([self.cfg.control.action_scale*self.cfg.control.hip_scale_reduction,
                                     self.cfg.control.action_scale,
                                     self.cfg.control.action_scale],device=self.device).repeat(1,4)
        
        actions_scaled = actions * action_scale
        # actions_scaled = torch.clip(actions_scaled,self.dof_pos_limits[:,0] - self.default_dof_pos,self.dof_pos_limits[:,1]-self.default_dof_pos)
        
        if self.cfg.control.safety_clip_actions:
            clip_idx = torch.logical_or(self.dof_pos < self.dof_pos_limits[:,0] + 0.087, self.dof_pos > self.dof_pos_limits[:,1] - 0.087).reshape(-1,12)
            if torch.any(clip_idx):
                actions_scaled[clip_idx] = torch.clip(actions_scaled,self.dof_pos_limits[:,0] - self.default_dof_pos,self.dof_pos_limits[:,1]-self.default_dof_pos)[clip_idx]

        self.actions_scaled  = actions_scaled.clone()
        if self.cfg.domain_rand.sim_latency:
            latency_buf = torch.ceil(self.episodic_latency / self.dt)
            actions_scaled[self.episode_length_buf < latency_buf] = 0.

        control_type = self.cfg.control.control_type
        if control_type=="P":

            if self.cfg.domain_rand.randomize_lag_timesteps:
                self.lag_buffer = self.lag_buffer[1:] + [actions_scaled.clone()]
                torques = self.p_gains*(self.lag_buffer[0] + self.default_dof_pos - self.dof_pos + self.motor_offsets) - self.d_gains*self.dof_vel
                # print(self.lag_buffer[0][0])
            elif self.cfg.domain_rand.sim_pd_latency:
                # Get the sampled latency
                idx_pd,sampled_latency_pd = self._sample_latency(pd=True)
                # Get the delayed observations:
                delayed_dof_pos, delayed_dof_vel = self._get_delayed_PD_states(idx_pd,sampled_latency_pd)

                torques = self.p_gains*(actions_scaled + self.default_dof_pos - delayed_dof_pos + self.motor_offsets) - self.d_gains*delayed_dof_vel
            else:
                # print(actions_scaled[0])
                torques = self.p_gains*(actions_scaled + self.default_dof_pos - self.dof_pos + self.motor_offsets) - self.d_gains*self.dof_vel

        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        elif control_type == "actuator_net":
            self.joint_pos_err = self.dof_pos - (actions_scaled + self.default_dof_pos) + self.motor_offsets
            self.joint_vel = self.dof_vel
            torques = self.actuator_network(self.joint_pos_err, self.joint_pos_err_last, self.joint_pos_err_last_last,
                                            self.joint_vel, self.joint_vel_last, self.joint_vel_last_last)
            self.joint_pos_err_last_last = torch.clone(self.joint_pos_err_last)
            self.joint_pos_err_last = torch.clone(self.joint_pos_err)
            self.joint_vel_last_last = torch.clone(self.joint_vel_last)
            self.joint_vel_last = torch.clone(self.joint_vel)
        elif control_type == "P_joint_pos":
            if self.cfg.domain_rand.sim_pd_latency:
                # Get the sampled latency
                idx_pd,sampled_latency_pd = self._sample_latency(pd=True)
                # Get the delayed observations:
                delayed_dof_pos, delayed_dof_vel = self._get_delayed_PD_states(idx_pd,sampled_latency_pd)

                torques = self.p_gains*(actions_scaled - delayed_dof_pos) - self.d_gains*delayed_dof_vel
            else:
                torques = self.p_gains*(actions_scaled - self.dof_pos) - self.d_gains*self.dof_vel
        
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        
        if self.cfg.domain_rand.randomize_motor_strength:
            torques *= self.motor_strengths

        return torch.clip(torques, -self.torque_limits, self.torque_limits)
    
    def _reset_spring_params(self,env_ids):
        '''
        Reset the spring parameters to their default values. If randomization is enabled, 
        add a random value from the range in the config file.
        '''
        # Get the default spring parameters:
        self.spring_stiffness[env_ids] = self.default_spring_stiffness[env_ids].clone()
        self.spring_damping[env_ids] = self.default_spring_damping[env_ids].clone()
        self.spring_rest_pos[env_ids] = self.default_spring_rest_pos[env_ids].clone()

        # If we want to randomize the spring parameters, add a random value from their range:
        if self.cfg.domain_rand.randomize_spring_params:

            K_variation = torch_rand_float(1. - self.cfg.domain_rand.ranges.spring_stiffness_percentage, 1. + self.cfg.domain_rand.ranges.spring_stiffness_percentage, (len(env_ids),3), device=self.device)
            D_variation = torch_rand_float(1. - self.cfg.domain_rand.ranges.spring_damping_percentage, 1. + self.cfg.domain_rand.ranges.spring_damping_percentage, (len(env_ids),3), device=self.device)
            rest_pos_variation = torch_rand_float(self.cfg.domain_rand.ranges.spring_rest_pos_range[0], self.cfg.domain_rand.ranges.spring_rest_pos_range[1], (len(env_ids),3), device=self.device)
            
            self.spring_stiffness[env_ids] *= K_variation.repeat(1,4)
            self.spring_damping[env_ids] *= D_variation.repeat(1,4)
            self.spring_rest_pos[env_ids] += rest_pos_variation.repeat(1,4)
            
    def _reset_state_history(self,env_ids):
        """ Resets state history of selected environments"""
        hist_len = self.cfg.env.state_history_length
        self.base_lin_vel_history[env_ids] = 0.0#self.base_lin_vel[env_ids,:].repeat(1,hist_len) 
        self.base_ang_vel_history[env_ids] = 0.0#self.base_ang_vel[env_ids,:].repeat(1,hist_len)
        self.root_states_history[env_ids] = 0.#self.root_states[env_ids,0:3].repeat(1,hist_len)
        self.dof_pos_history[env_ids] = 0.0#self.default_dof_pos.repeat(1,hist_len)
        self.dof_vel_history[env_ids] = 0.0#self.dof_vel[env_ids,:].repeat(1,hist_len)
        self.actions_history[env_ids] = 0.
        self.contacts_history[env_ids] = 0.0#self.contacts[env_ids,:].repeat(1,hist_len)
        self.base_quat_history[env_ids] = torch.tensor([0.,0.,0.,1.],device=self.device).repeat(len(env_ids),hist_len)#self.root_states[env_ids,3:7].repeat(1,hist_len)
        self.ori_error_history[env_ids] = 0.
        # self.error_quat_history[env_ids] = 0.#quat_distance(self.root_states[env_ids,3:7],self.commands[env_ids, 3:7],as_quat=True).repeat(1,hist_len)

    def _reset_stored_states(self,env_ids):
        """ Resets stored states of selected environments"""
        self.base_lin_vel_stored[env_ids] = self.base_lin_vel[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.base_ang_vel_stored[env_ids] = self.base_ang_vel[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.root_states_stored[env_ids] = self.root_states[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.dof_pos_stored[env_ids] = self.dof_pos[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.dof_vel_stored[env_ids] = 0.
        self.actions_stored[env_ids] = 0.
        self.contacts_stored[env_ids] = self.contacts[env_ids,:].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.base_quat_stored[env_ids] = self.base_quat[env_ids].repeat(1,self.cfg.env.state_history_length).unsqueeze(-1)
        self.ori_error_stored[env_ids] = 0
        # self.error_quat_stored[env_ids] = 0
        self.force_sensor_stored[env_ids] = 0

        self.pd_dof_pos_stored[env_ids] = self.dof_pos[env_ids,:].unsqueeze(-1)
        self.pd_dof_vel_stored[env_ids] = self.dof_vel[env_ids,:].unsqueeze(-1)
        
        
     
    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        # For continuous jumping only reset agents that have been terminated
        # or some of the ones that have finished the episode (with given probability)
        if self.cfg.env.continuous_jumping:
            env_ids = self.cont_jump_reset_env_ids
            if len(env_ids)==0:
                return
            

        
        des_foot_poses = torch.zeros((self.num_envs,4,3),device=self.device)
        des_foot_poses[:] = torch.tensor(self.cfg.init_state.rel_foot_pos).to(self.device).transpose(1,0).view(1,4,3)
        des_foot_poses[env_ids,:,:2] += torch_rand_float(-0.05, 0.05, (len(env_ids),8), device=self.device).reshape(-1,4,2)
        des_foot_poses[env_ids,:,2] += torch_rand_float(-0.08, 0.08, (len(env_ids),1), device=self.device)
        # Add random shift of the base pos:
        base_pos_shift = torch_rand_float(-0.02, 0.02, (len(env_ids),2), device=self.device)
        des_foot_poses[env_ids,:,:2] += base_pos_shift.unsqueeze(1)
        
        if self.cfg.domain_rand.randomize_dof_pos:
            self.base_pos_IK[env_ids,2] = (-des_foot_poses[env_ids,:,2]).max(dim=1)[0]
            self.base_pos_IK[env_ids,:2] = base_pos_shift.clone()
            self.dof_pos[env_ids] = self.inverse_kinematics_tensor(des_foot_poses[env_ids],env_ids)
            # self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.7, 1.3, (len(env_ids), self.num_dof), device=self.device)
        else:
            self.dof_pos[env_ids] = self.default_dof_pos

        self.dof_vel[env_ids] = 0.
        

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                            gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        
        # For continuous jumping only reset agents that have been terminated
        # or some of the ones that have finished the episode (with given probability)
        env_ids_all = env_ids.clone()
        self.initial_root_states[env_ids_all,:] = self.root_states[env_ids_all,:].clone()
        if self.cfg.env.continuous_jumping:
            env_ids = self.cont_jump_reset_env_ids
            if len(env_ids)==0:
                return
            

        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state

            self.root_states[env_ids, 0] += self.env_origins[env_ids,0]
            self.root_states[env_ids, 1] += self.env_origins[env_ids,1]

            # self.root_states[env_ids, 2] += self.env_origins[env_ids,2]
            # self.root_states[env_ids, 1] += torch_rand_float(-.1, .1, (len(env_ids), 2), device=self.device)[:,1] # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]


        if self.cfg.domain_rand.randomize_dof_pos:
            self.root_states[env_ids, :2] += self.base_pos_IK[env_ids,:2]
            self.root_states[env_ids, 2] = self.base_pos_IK[env_ids,2]

       # When jumping on objects randomize the initial position of the robot instead of the obstacle
        # (for computational efficiency)
        if self.cfg.commands.randomize_commands and self.jump_type == "forward_with_obstacles":#self.cfg.terrain.mesh_type=='trimesh':
            box_widths = self.env_properties[env_ids,0]
            # Shift robot init pos by the half width of the obstacle + 0.1881 (half of the robot width)
            # So that the foot starts right at the obstacle
            self.root_states[env_ids, 0] -= (box_widths/2 + 0.1881)
            # And then shift it further by a random amount
            self.root_states[env_ids, 0] += torch_rand_float(-0.25, -0.2, (len(env_ids), 1), device=self.device).flatten()
            # Also add random component to y position
            self.root_states[env_ids, 1] += torch_rand_float(-0.1, 0.1, (len(env_ids), 1), device=self.device).flatten()

        else:
            # Add random shift of the base pos:
            self.root_states[env_ids, :2] += torch_rand_float(-0.5, 0.5, (len(env_ids), 2), device=self.device)

        # Reset the base position and velocity to different values:
        # First sample whether to randomise or not:
        randomize_pos_vel = self.pos_vel_randomisation_dist.sample()[env_ids]
        
        self.initial_root_states_nonrandomised[env_ids_all,:] = self.root_states[env_ids_all,:].clone()

        env_ids_randomize = env_ids[randomize_pos_vel==1]

        self._pos_vel_rand_envs[env_ids] = 0
        self._pos_vel_rand_envs[env_ids_randomize] = 1

        if self.cfg.domain_rand.randomize_robot_pos:
            self.root_states[env_ids_randomize, 0] += torch_rand_float(self.pos_variation[0,0], self.pos_variation[1,0], (len(env_ids_randomize), 1), device=self.device).flatten()
            self.root_states[env_ids_randomize, 1] += torch_rand_float(self.pos_variation[0,1], self.pos_variation[1,1], (len(env_ids_randomize), 1), device=self.device).flatten()
            self.root_states[env_ids_randomize, 2] += torch_rand_float(self.pos_variation[0,2], self.pos_variation[1,2], (len(env_ids_randomize), 1), device=self.device).flatten()
 
        if self.cfg.domain_rand.randomize_robot_vel:
            self.root_states[env_ids_randomize, 7] = torch_rand_float(self.vel_variation[0,0], self.vel_variation[1,0], (len(env_ids_randomize), 1), device=self.device).flatten()
            self.root_states[env_ids_randomize, 8] = torch_rand_float(self.vel_variation[0,1], self.vel_variation[1,1], (len(env_ids_randomize), 1), device=self.device).flatten()
            self.root_states[env_ids_randomize, 9] = torch_rand_float(self.vel_variation[0,2], self.vel_variation[1,2], (len(env_ids_randomize), 1), device=self.device).flatten()

        # Reset the orientation:
        if self.cfg.domain_rand.randomize_robot_ori:

            r =  torch_rand_float(self.ori_variation[0,0],self.ori_variation[1,0], (len(env_ids), 1), device=self.device).reshape(-1)
            p =  torch_rand_float(self.ori_variation[0,1],self.ori_variation[1,1], (len(env_ids), 1), device=self.device).reshape(-1)
            y =  torch_rand_float(self.ori_variation[0,2],self.ori_variation[1,2], (len(env_ids), 1), device=self.device).reshape(-1)
            
           
            self.root_states[env_ids, 3:7] = quat_from_euler_xyz(r,p,y)

    

        # Record the initial root states for ALL reset agents (even for continuous jumping)
        # as it is needed to compute the relative distances for the jumps.
        self.initial_root_states[env_ids_all,:] = self.root_states[env_ids_all,:].clone()
        self.initial_foot_poses[env_ids_all] = self.feet_pos[env_ids_all].clone()

        
        # 
         
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots_upwards(self,env_ids):
        
        self.root_states[env_ids,9] = torch_rand_float(3.0, 5.0, (self.num_envs, 1), device=self.device).flatten()[env_ids]
        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots_desired(self,env_ids):
        """ Randomly pushes some robots towards the goal just before takeoff. Emulates an impulse by setting a randomized base velocity. 
        """
        
        random_push = torch.randint(0,10,(self.num_envs,1),device=self.device).squeeze()
        if self.jump_type == "forward_with_obstacles":
            return
            # idx = torch.logical_and(random_push<4,env_ids)
        else:
            idx = torch.logical_and(random_push<8,env_ids)

        des_vel = self.command_vels[idx,:2]
        des_yaw_vel = self.command_vels[idx,5]
        self.root_states[idx, 7:9] = des_vel #+ torch_rand_float(-0.3, 0.3, (self.num_envs, 2), device=self.device)[idx] # lin vel x/y
        # self.root_states[idx, 7:9] = torch.clip(self.root_states[idx, 7:9],min=0.0)
        # self.root_states[idx, 9] += torch_rand_float(0.1, 0.5, (self.num_envs, 1), device=self.device).flatten()[idx]
        self.root_states[idx, 12] = des_yaw_vel

        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        env_ids = self.has_jumped.clone()
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[env_ids, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device)[env_ids] # lin vel x/y
        
        self.root_states[self.mid_air, 7:9] += torch_rand_float(-max_vel/2, max_vel/2, (self.num_envs, 2), device=self.device)[self.mid_air] # lin vel x/y


        
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        # 
        success_rate = torch.sum(self.success_rate[env_ids], dim=1)/(self.success_rate.shape[-1])
        # robots that were sucessful for enough trials should be moved to harder jumps:
        # Currently if it succeeds once it's moved up
        move_up = torch.logical_and(success_rate > 0.0, torch.all(self.success_rate[env_ids] >= 0.))
        # robots that were not successful for enough trials should be moved to easier jumps:
        # if it fails in both trials it's moved down
        move_down = torch.logical_and(success_rate <= 0.0, torch.all(self.success_rate[env_ids] >= 0.)) * ~move_up
        self.terrain_levels[env_ids] += 1 * move_up #- 1 * move_down
        self.terrain_levels[env_ids] -= 1 * move_down
        # Reset success rate for robots that have changed difficulty levels
        self.success_rate[env_ids[torch.logical_or(move_up,move_down)]] = -1.
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
        self.env_properties[env_ids] = self.terrain_properties[self.terrain_levels[env_ids], self.terrain_types[env_ids]]

    def _update_domain_rand_curriculum(self,env_ids):
        if len(env_ids)==0:
            return
        idx = env_ids[self.max_height[env_ids]>=0.8]
        self.pos_vel_randomisation_prob[idx] = torch.clamp(self.pos_vel_randomisation_prob[idx] - 0.01,min=0.0) 
        self.pos_vel_randomisation_dist = torch.distributions.bernoulli.Bernoulli(self.pos_vel_randomisation_prob)
    def _update_reward_curriculum(self):
        """
        Implements a curriculum of decreasing sigma for the negative reward

        Args:
            None
        """
        
        if self.common_step_counter < (self.cfg.rewards.sigma_neg_rew_initial_duration / 5 * self.max_episode_length):
            return
        # Approx 5 episodes per update step:
        episodes_num = (self.cfg.rewards.sigma_neg_rew_curriculum_duration / 5)
        min_sigma = 0.05
        # Determine the size of the step based on initial and min sigma values, and number of episodes:
        stepsize = (self.cfg.rewards.sigma_rew_neg - min_sigma) / episodes_num

        if (self.common_step_counter -1) % (self.max_episode_length)==0:

            self.sigma_rew_neg = np.clip(self.sigma_rew_neg-stepsize,a_min=min_sigma,a_max=self.cfg.rewards.sigma_rew_neg)

    def _update_command_curriculum(self,env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """

        # Update the possible range of command distance variation:

        if not self.init_done:
            # don't change on initial reset
            return
        # 
        success_rate = torch.sum(self.success_rate[env_ids], dim=1)/(self.success_rate.shape[-1])
        # robots that were sucessful for enough trials should be moved to harder jumps:
        # Currently if it succeeds once it's moved up
        move_up = torch.logical_and(success_rate > 0.0, torch.all(self.success_rate[env_ids] >= 0.))
        # robots that were not successful for enough trials should be moved to easier jumps:
        # if it fails in both trials it's moved down
        move_down = torch.logical_and(success_rate <= 0.0, torch.all(self.success_rate[env_ids] >= 0.)) * ~move_up
       
        self.command_dist_levels[env_ids] += 1 * move_up #- 1 * move_down
        self.command_dist_levels[env_ids] -= 1 * move_down
        

        max_command_dist_level = self.cfg.commands.num_levels - 1

        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] >= max_command_dist_level/2)] -= 1 * move_up
        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] >= max_command_dist_level/2)] += 1 * move_down
        self.reset_landing_error[env_ids * (self.command_dist_levels[env_ids] < max_command_dist_level/2)] = max_command_dist_level
        

        self.command_dist_levels = torch.clip(self.command_dist_levels, min=0, max=max_command_dist_level) # (the minumum level is zero)
        self.reset_landing_error = torch.clip(self.reset_landing_error, min=0, max=max_command_dist_level)
        # Reset success rate for robots that have changed difficulty levels
        self.success_rate[env_ids[torch.logical_or(move_up,move_down)]] = -1.

        # Robots that solve the last level are sent to a random one
        # self.command_dist_levels[env_ids] = torch.where(self.command_dist_levels[env_ids]>=max_command_dist_level,
        #                                            torch.randint_like(self.command_dist_levels[env_ids], max_command_dist_level),
        #                                            torch.clip(self.command_dist_levels[env_ids], 0)) # (the minumum level is zero)
        

    def _compute_observation_vector_length(self,state_history_length):


        num_observations += 2*3*state_history_length + 3 + 3*12*state_history_length
        
        # + orientation (quaternion*state_history_length)
        if self.cfg.env.known_quaternion:
            num_observations += 4*state_history_length
        if self.cfg.env.known_ori_error:
            num_observations += 1*state_history_length
        # if self.cfg.env.known_error_quaternion:
        #     num_observations += 4*state_history_length
        # And and height if it's known (1*state_history_length)
        if self.cfg.env.known_height:
            num_observations += 1*state_history_length
        # And desired jumping position (relative) and quat if added:
        if self.cfg.env.jumping_target:
            num_observations += 7 + 6

        if self.cfg.env.pass_remaining_time: # Pass remaining time in the episode (one scalar value)
            num_observations += 1

        if self.cfg.env.pass_has_jumped: # Pass has_jumped bool (one scalar value)
            num_observations += 1
        
        if self.cfg.env.known_contact_feet: # Pass contact state at the feet (4*state_history_length)
            num_observations += 4*state_history_length

        return num_observations

    def _get_noise_scale_vec_without_history(self,cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        hist_len = self.cfg.env.state_history_length
        noise_vec[:3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3*hist_len:3+3*hist_len] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[6*hist_len:3+6*hist_len] = noise_scales.gravity * noise_level
        noise_vec[6*hist_len:12+6*hist_len] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18*hist_len:12+18*hist_len] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[30*hist_len:12 + 30*hist_len] = 0. # previous actions

        # Add noise to the additional observations:
        obs_index_append = 42*hist_len
        if self.cfg.env.known_quaternion:
            noise_vec[obs_index_append:obs_index_append+4] = noise_scales.quat * noise_level * self.obs_scales.quat
            obs_index_append += 4*hist_len
        if self.cfg.env.known_ori_error:
            noise_vec[obs_index_append:obs_index_append+1] = noise_scales.ori_error * noise_level * self.obs_scales.ori_error
            obs_index_append += hist_len
        # if self.cfg.env.known_error_quaternion:
        #     noise_vec[obs_index_append:obs_index_append+4] = noise_scales.error_quat * noise_level * self.obs_scales.error_quat
        #     obs_index_append += 4*hist_len
        if self.cfg.env.jumping_target:
            noise_vec[obs_index_append:obs_index_append+13] = 0.
            obs_index_append += 13
        if self.cfg.env.known_height:
            noise_vec[obs_index_append:obs_index_append+1] = noise_scales.height * noise_level * self.obs_scales.height
            obs_index_append += 1*hist_len
        if self.cfg.env.pass_remaining_time:
            noise_vec[obs_index_append:obs_index_append+1] = 0
            obs_index_append += 1
        if self.cfg.env.pass_has_jumped:
            noise_vec[obs_index_append:obs_index_append+1] = 0
            obs_index_append += 1
        if self.cfg.env.known_contact_feet:
            noise_vec[obs_index_append:obs_index_append+4] =  noise_scales.contacts * noise_level * self.obs_scales.contacts
            obs_index_append += 4*hist_len
        if self.cfg.terrain.measure_heights:
            noise_vec[obs_index_append:obs_index_append+187] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        
        return noise_vec

    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        hist_len = self.cfg.env.state_history_length
        noise_vec[:3*hist_len] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        noise_vec[3*hist_len:6*hist_len] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        # noise_vec[6*hist_len:3+6*hist_len] = noise_scales.gravity * noise_level
        noise_vec[6*hist_len:18*hist_len] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[18*hist_len:30*hist_len] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[30*hist_len:42*hist_len] = 0. # previous actions

        # Add noise to the additional observations:
        obs_index_append = 42*hist_len
        if self.cfg.env.known_quaternion:
            noise_vec[obs_index_append:obs_index_append+4*hist_len] = noise_scales.quat * noise_level * self.obs_scales.quat
            obs_index_append += 4*hist_len
        if self.cfg.env.known_ori_error:
            noise_vec[obs_index_append:obs_index_append+hist_len] = noise_scales.ori_error * noise_level * self.obs_scales.ori_error
            obs_index_append += hist_len
        # if self.cfg.env.known_error_quaternion:
        #     noise_vec[obs_index_append:obs_index_append+4*hist_len] = noise_scales.error_quat * noise_level * self.obs_scales.error_quat
        #     obs_index_append += 4*hist_len
        if self.cfg.env.jumping_target:
            noise_vec[obs_index_append:obs_index_append+7] = 0.
            obs_index_append += 13
        if self.cfg.env.known_height:
            noise_vec[obs_index_append:obs_index_append+hist_len] = noise_scales.height * noise_level * self.obs_scales.height
            obs_index_append += 1*hist_len
        if self.cfg.env.pass_remaining_time:
            noise_vec[obs_index_append:obs_index_append+1] = 0
            obs_index_append += 1
        if self.cfg.env.pass_has_jumped:
            noise_vec[obs_index_append:obs_index_append+1] = 0
            obs_index_append += 1
        if self.cfg.env.known_contact_feet:
            noise_vec[obs_index_append:obs_index_append+4*hist_len] =  noise_scales.contacts * noise_level * self.obs_scales.contacts
            obs_index_append += 4*hist_len
        if self.cfg.terrain.measure_heights:
            noise_vec[obs_index_append:obs_index_append+187] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        
        
        return noise_vec

    #----------------------------------------
    def _init_custom_buffers__(self):
        # domain randomization properties
        self.friction_coeffs = self.default_friction * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.restitutions = self.default_restitution * torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device,
                                                                  requires_grad=False)
        self.joint_friction_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        self.joint_damping_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        self.joint_armatures = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device,requires_grad=False)
        self.payload_masses = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.link_masses = torch.zeros(self.num_envs, self.num_bodies-1, dtype=torch.float, device=self.device,requires_grad=False)
        self.com_displacements = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,
                                             requires_grad=False)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                          requires_grad=False)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, dtype=torch.float, device=self.device,
                                         requires_grad=False)
        self.p_gains = torch.zeros((self.num_envs,self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros((self.num_envs,self.num_actions), dtype=torch.float, device=self.device, requires_grad=False)
        
        self.gravities = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device,requires_grad=False)

        
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)
        
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        _fsdata = self.gym.acquire_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.rigid_body_state = self.rb_states.view(self.num_envs,-1, 13) #[num_envs,num_bodies,13]
        self.force_sensor_readings = gymtorch.wrap_tensor(_fsdata).view(self.num_envs,5,6)[:,:4,:3]
        self.imu_tensor = gymtorch.wrap_tensor(_fsdata).view(self.num_envs,5,6)[:,4,:3]
        self.dof_acc = torch.zeros_like(self.dof_vel)
        self.last_dof_acc = torch.zeros_like(self.dof_acc)
        self.dof_jerk = torch.zeros_like(self.dof_acc)
        self.euler = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.des_angles_euler = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)
        self.base_lin_vel_imu = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device, requires_grad=False)

        self.rigid_body_state = self.rb_states.view(self.num_envs,-1, 13) #[num_envs,num_bodies,13]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # create some curriculum data:
        self.pos_variation_ini = torch.tensor([self.cfg.domain_rand.ranges.min_robot_pos,self.cfg.domain_rand.ranges.max_robot_pos], dtype=torch.float, device=self.device, requires_grad=False)
        self.pos_variation = self.pos_variation_ini
        self.vel_variation_ini = torch.tensor([self.cfg.domain_rand.ranges.min_robot_vel,self.cfg.domain_rand.ranges.max_robot_vel], dtype=torch.float, device=self.device, requires_grad=False)
        self.vel_variation = self.vel_variation_ini

        self.ori_variation = torch.tensor([self.cfg.domain_rand.ranges.min_ori_euler,self.cfg.domain_rand.ranges.max_ori_euler], dtype=torch.float, device=self.device, requires_grad=False)
        self.curriculum_increment = 0
        self.command_curriculum_iter = 0
        # Desired jumping position randomisation variables:
        self.pos_command_variation_increment =  torch.tensor(self.cfg.commands.ranges.pos_variation_increment,device=self.device)

        self.pos_command_variation_limits = torch.zeros((2,3),device=self.device)
        self.pos_command_variation_ini = torch.zeros((2,3),device=self.device)

        self.pos_command_variation_limits[0,:] = torch.tensor([self.cfg.commands.ranges.pos_dx_lim[0],self.cfg.commands.ranges.pos_dy_lim[0],self.cfg.commands.ranges.pos_dz_lim[0]])
        self.pos_command_variation_limits[1,:] = torch.tensor([self.cfg.commands.ranges.pos_dx_lim[1],self.cfg.commands.ranges.pos_dy_lim[1],self.cfg.commands.ranges.pos_dz_lim[1]])
        self.pos_command_variation_ini[0,:] = torch.tensor([self.cfg.commands.ranges.pos_dx_ini[0],self.cfg.commands.ranges.pos_dy_ini[0],self.cfg.commands.ranges.pos_dz_ini[0]])
        self.pos_command_variation_ini[1,:] = torch.tensor([self.cfg.commands.ranges.pos_dx_ini[1],self.cfg.commands.ranges.pos_dy_ini[1],self.cfg.commands.ranges.pos_dz_ini[1]])

        self.pos_command_variation = torch.zeros((2,3),device=self.device)
        # Initially set the desired jumping position variation to the initial one from cfg.
        self.pos_command_variation = self.pos_command_variation_ini.clone()

        num_levels = self.cfg.commands.num_levels
        self.command_dist_levels = torch.randint(0, num_levels, (self.num_envs,1), device=self.device).flatten()
        self.reset_landing_error = torch.zeros_like(self.command_dist_levels)
        self.memory_log = 0
        # Spring stuff:
        # Default spring parameters:
        self.default_spring_stiffness, self.default_spring_damping, self.default_spring_rest_pos = self._get_spring_params()
        # Spring parameters that will be used in the simulation (might be updated if using randomisation of
        # spring parameters):
        self.spring_stiffness = self.default_spring_stiffness.clone()
        self.spring_damping = self.default_spring_damping.clone()
        self.spring_rest_pos = self.default_spring_rest_pos.clone()



        # initialize some data used later on
        self.common_step_counter = 0
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.noise_scale_vec_no_history = self._get_noise_scale_vec_without_history(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.torques_to_apply = torch.zeros_like(self.torques)
        self.torques_springs = torch.zeros_like(self.torques)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions_scaled = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_feet_vel = torch.zeros(self.num_envs,4,3, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_acc_prev = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.max_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.min_height = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.tracking_error_store = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.tracking_error_percentage_store = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) 
        self.command_vels = torch.zeros(self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, z vel
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.contact_filt_prev = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.was_in_flight = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.mid_air = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.not_pushed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.settled_after_init = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.feet_pos = torch.zeros(self.num_envs, len(self.feet_indices), 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        self.contacts = torch.ones(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        self.ori_error = torch.zeros(self.num_envs, 1,dtype=torch.float, device=self.device, requires_grad=False)
        # self.error_quat = torch.zeros(self.num_envs, 4,dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = 0
        self.initial_root_states = torch.zeros_like(self.root_states)
        self.initial_root_states_nonrandomised = torch.zeros_like(self.root_states)
        self.initial_foot_poses = torch.zeros_like(self.rigid_body_state[:,self.feet_indices,:3])
        self.landing_poses = torch.zeros(self.num_envs, 7, dtype=torch.float, device=self.device, requires_grad=False)
        self.landing_foot_poses = torch.zeros(self.num_envs, 4, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.reset_idx_landing_error = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.continuous_jump_reset_prob = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.cont_jump_reset_env_ids = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        
        self.additional_termination_conditions = True

        max_ep_len = self.max_episode_length.astype(int)
        self.mean_dof_acc_stored = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean_base_acc_stored = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean_action_rate_stored = torch.zeros(1,self.num_envs,self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.mean_action_rate_2_stored = torch.zeros(1, dtype=torch.float, device=self.device, requires_grad=False)

        self.pos_vel_randomisation_prob = self.cfg.domain_rand.pos_vel_random_prob + torch.zeros(self.num_envs,dtype=torch.float,device=self.device)
        if not self.cfg.domain_rand.randomize_robot_pos and not self.cfg.domain_rand.randomize_robot_vel:
            self.pos_vel_randomisation_prob = torch.zeros(self.num_envs,dtype=torch.float,device=self.device)
        self.pos_vel_randomisation_dist = torch.distributions.bernoulli.Bernoulli(self.pos_vel_randomisation_prob)
        
        up_jump_prob = self.cfg.commands.upward_jump_probability
        self.up_jump_distribution = torch.distributions.bernoulli.Bernoulli(torch.tensor([up_jump_prob],device=self.device))
        self.push_upwards_distr = torch.distributions.bernoulli.Bernoulli(torch.tensor([self.cfg.domain_rand.push_upwards_prob],device=self.device))
        self.success_rate = -torch.ones(self.num_envs, 2, dtype=torch.float, device=self.device, requires_grad=False)

        self.jump_type = self.cfg.env.jump_type
        self.num_steps_before_termination = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)

        has_jumped_randomisation_prob = self.cfg.domain_rand.has_jumped_random_prob
        self.has_jumped_randomisation_prob = torch.distributions.bernoulli.Bernoulli(torch.tensor([has_jumped_randomisation_prob],device=self.device))
        self._has_jumped_rand_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.has_jumped_reset_flag = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.05],device=self.device))
        self._reset_randomised_has_jumped_timer = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self._has_jumped_switched_time = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.settled_after_init_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device, requires_grad=False)
        self.base_pos_IK = torch.zeros(self.num_envs,3, dtype=torch.float, device=self.device, requires_grad=False)
        self.push_upwards_timer = torch.zeros(self.num_envs, dtype=torch.int, device=self.device, requires_grad=False)
        self.push_upwards_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._pos_vel_rand_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        # State history:
        if self.cfg.env.use_state_history:
            self.base_lin_vel_history = torch.zeros(self.num_envs, self.base_lin_vel.shape[-1]*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.base_ang_vel_history = torch.zeros(self.num_envs, self.base_ang_vel.shape[-1]*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.root_states_history = torch.zeros(self.num_envs, self.root_states.shape[-1]*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_pos_history = torch.zeros(self.num_envs, self.num_dof*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_history = torch.zeros(self.num_envs, self.num_dof*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.actions_history = torch.zeros(self.num_envs, self.num_actions*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.contacts_history = torch.zeros(self.num_envs, len(self.feet_indices)*self.cfg.env.state_history_length, dtype=torch.bool, device=self.device, requires_grad=False)
            self.base_quat_history = torch.zeros(self.num_envs, 4*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.ori_error_history = torch.zeros(self.num_envs, self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            # self.error_quat_history = torch.zeros(self.num_envs, 4*self.cfg.env.state_history_length, dtype=torch.float, device=self.device, requires_grad=False)
            self.has_jumped_history = torch.zeros(self.num_envs, self.cfg.env.state_history_length, dtype=torch.bool, device=self.device, requires_grad=False)
        if self.cfg.env.pass_remaining_time:
            self.remaining_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # Store the last N states at a higher rate for PD control (PD latency)
        self.pd_dof_pos_stored = torch.zeros(self.num_envs, self.num_dof,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.pd_dof_vel_stored = torch.zeros(self.num_envs, self.num_dof,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)

        self.lag_buffer = [torch.zeros_like(self.dof_pos) for i in range(self.cfg.domain_rand.lag_timesteps+1)]


        # Store the last N states for each env (use to simulate latency)
        self.base_lin_vel_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*self.base_lin_vel.shape[-1],self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_ang_vel_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*self.base_ang_vel.shape[-1],self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.root_states_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*13,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.root_states_stored[:,:,:] = self.root_states.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*13,1)
        self.dof_pos_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*self.num_dof,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_pos_stored[:,:,:] = self.dof_pos.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*self.num_dof,1)
        self.dof_vel_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*self.num_dof,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.dof_vel_stored[:,:,:] = self.dof_vel.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*self.num_dof,1)
        self.actions_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*self.num_actions,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.contacts_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*len(self.feet_indices),self.cfg.env.state_stored_length, dtype=torch.bool, device=self.device, requires_grad=False)
        self.contacts_stored[:,:,:] = self.contacts.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*len(self.feet_indices),1)
        self.base_quat_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*4,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.base_quat_stored[:,:,:] =  self.base_quat.repeat(1,self.cfg.env.state_history_length).view(self.num_envs, self.cfg.env.state_history_length*4,1)
        self.ori_error_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*1, self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        # self.error_quat_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*4,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.force_sensor_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*4, 3,self.cfg.env.state_stored_length, dtype=torch.float, device=self.device, requires_grad=False)
        self.has_jumped_stored = torch.zeros(self.num_envs, self.cfg.env.state_history_length*1,self.cfg.env.state_stored_length, dtype=torch.bool, device=self.device, requires_grad=False)

        self.base_lin_vel_delayed = torch.zeros_like(self.base_lin_vel_history)
        self.base_ang_vel_delayed = torch.zeros_like(self.base_ang_vel_history)
        self.root_states_delayed = torch.zeros_like(self.root_states_history)
        self.dof_pos_delayed = torch.zeros_like(self.dof_pos_history)
        self.dof_vel_delayed = torch.zeros_like(self.dof_vel_history)
        self.actions_delayed = torch.zeros_like(self.actions_history)
        self.contacts_delayed = torch.zeros_like(self.contacts_history)
        self.base_quat_delayed = torch.zeros_like(self.base_quat_history)
        self.ori_error_delayed = torch.zeros_like(self.ori_error_history)
        # self.error_quat_delayed = torch.zeros_like(self.error_quat_history)
        # self.force_sensor_delayed = torch.zeros_like(self.force_sensor_readings)
        self.has_jumped_delayed = torch.zeros_like(self.has_jumped_history) 

        self.latency_range = self.cfg.domain_rand.ranges.latency_range
        self.pd_latency_range = self.cfg.domain_rand.ranges.pd_latency_range

        self.episodic_pd_latency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.episodic_latency = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        self.prev_sampled_latency = self.cfg.domain_rand.ranges.latency_range[1] + torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)


        self.sigma_rew_neg = self.cfg.rewards.sigma_rew_neg
        self.reward_logs = {}

        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[:,i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[:,i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[:,i] = 0.
                self.d_gains[:,i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)

        if self.cfg.control.control_type == "actuator_net":
            actuator_path = f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/../../resources/actuator_nets/unitree_go1.pt'
            actuator_network = torch.jit.load(actuator_path).to(self.device)

            def eval_actuator_network(joint_pos, joint_pos_last, joint_pos_last_last, joint_vel, joint_vel_last,
                                      joint_vel_last_last):
                xs = torch.cat((joint_pos.unsqueeze(-1),
                                joint_pos_last.unsqueeze(-1),
                                joint_pos_last_last.unsqueeze(-1),
                                joint_vel.unsqueeze(-1),
                                joint_vel_last.unsqueeze(-1),
                                joint_vel_last_last.unsqueeze(-1)), dim=-1)
                torques = actuator_network(xs.view(self.num_envs * 12, 6))
                return torques.view(self.num_envs, 12)

            self.actuator_network = eval_actuator_network

            self.joint_pos_err_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_pos_err_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last_last = torch.zeros((self.num_envs, 12), device=self.device)
            self.joint_vel_last = torch.zeros((self.num_envs, 12), device=self.device)

        if self.cfg.control.use_action_filter:
            if self.cfg.control.filter_type == "butterworth":
                self.action_filter = ButterworthFilter(order=self.cfg.control.butterworth_order,f_cutoff=self.cfg.control.filter_freq,
                                                    sampling_rate=1/(self.dt),
                                                    q0 = self.default_dof_pos,
                                                    num_envs=self.num_envs,
                                                    num_joints=self.num_dof,
                                                    device = self.device)
            elif self.cfg.control.filter_type == "EMA":
                # self.action_filter = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
                self.action_filter = EMAFilter(f_cutoff=self.cfg.control.filter_freq,
                                            sampling_rate=1/(self.dt),
                                            q0 = self.default_dof_pos,
                                            num_envs=self.num_envs,
                                            num_joints=self.num_dof,
                                            device = self.device)



    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            self.reward_logs[name] = []
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))


        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)


    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity
        asset_options.use_physx_armature = self.cfg.asset.use_physx_armature
        # asset_options.override_inertia = True
        # asset_options.override_com = True
        asset_options.enable_gyroscopic_forces = True
        # asset_options.vhacd_enabled = True
        # asset_options.vhacd_params.resolution = 300000
        # asset_options.vhacd_params.max_convex_hulls = 10
        # asset_options.vhacd_params.max_num_vertices_per_ch = 64


        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names) #
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])
        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []

        feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            feet_indices[i] = self.gym.find_asset_rigid_body_index(robot_asset, feet_names[i])
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.enable_forward_dynamics_forces = False # for example gravity
        sensor_props.enable_constraint_solver_forces = True # for example contacts
        sensor_props.use_world_frame = True

        for feet_idx in feet_indices:
            sensor_pose = gymapi.Transform()
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose,sensor_props)
        
        # Add imu sensor:
        body_idx = self.gym.find_asset_rigid_body_index(robot_asset, "base")
        sensor_pose = gymapi.Transform()#gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
        sensor_props.enable_forward_dynamics_forces = True # for example gravity
        sensor_props.enable_constraint_solver_forces = True # for example contacts
        sensor_props.use_world_frame = True
        self.gym.create_asset_force_sensor(robot_asset, body_idx, sensor_pose,sensor_props)

        self.default_friction = rigid_shape_props_asset[1].friction
        self.default_restitution = rigid_shape_props_asset[1].restitution
        self._init_custom_buffers__()
        self.randomize_rigid_body_props(torch.arange(self.num_envs, device=self.device))
        self.randomize_dof_props(torch.arange(self.num_envs, device=self.device))
        self._randomize_gravity()

        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        if not self.headless and self.cfg.viewer.simulate_camera:
            camera_props = gymapi.CameraProperties()
            camera_props.width = 1920
            camera_props.height = 1080
            self.camera_handle = self.gym.create_camera_sensor(self.envs[0], camera_props)

        self._refresh_actor_rigid_shape_props(torch.arange(self.num_envs, device=self.device))
        self._refresh_actor_dof_props(torch.arange(self.num_envs, device=self.device))

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])



    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            self.env_properties = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            # max_init_level = self.cfg.terrain.max_init_terrain_level
            # if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.terrain_properties = torch.from_numpy(self.terrain.env_properties).to(self.device).to(torch.float)
            self.env_properties[:] = self.terrain_properties[self.terrain_levels, self.terrain_types]
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        print(self.dt)
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        # self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        self.command_distances = class_to_dict(self.cfg.commands.distances)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

        self.cfg.domain_rand.gravity_rand_interval = np.ceil(self.cfg.domain_rand.gravity_rand_interval_s / self.dt)
        self.cfg.domain_rand.gravity_rand_duration = np.ceil(
            self.cfg.domain_rand.gravity_rand_interval * self.cfg.domain_rand.gravity_impulse_duration)

    def _draw_debug_goal(self):
        """ Draws desired goal points for debugging.
            Default behaviour: draws goal position
        """
        
        self.gym.clear_lines(self.viewer)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        sphere_geom_start = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(0, 1, 0))

        for i in range(self.num_envs):
            goal_pos = (self.initial_root_states[i, :3] + self.commands[i,0:3]).cpu().numpy()
            goal_pos[2] = self.commands[i,2]
            # goal_pos = self.env_origins[i].clone()
            # Plot the xy position as a sphere on the ground plane
            sphere_pose = gymapi.Transform(gymapi.Vec3(goal_pos[0], goal_pos[1], goal_pos[2]), r=None)
            gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

            sphere_pose_start = gymapi.Transform(gymapi.Vec3(self.initial_root_states[i, 0], self.initial_root_states[i, 1], 0.0), r=None)
            gymutil.draw_lines(sphere_geom_start, self.gym, self.viewer, self.envs[i], sphere_pose_start)
            # Draw line to the goal
            if self.cfg.env.debug_draw_line_goal:
                ini_pos = (self.initial_root_states[i, :3]).cpu().numpy()
                verts = np.empty((1, 2), dtype=gymapi.Vec3.dtype)
                verts[0][0] = (ini_pos[0], ini_pos[1],ini_pos[2])
                verts[0][1] = (goal_pos[0], goal_pos[1],goal_pos[2])
                colors = np.empty(1, dtype=gymapi.Vec3.dtype)
                colors[0] = (0,1,0)
                self.gym.add_lines(self.viewer, self.envs[i], 3, verts, colors)


    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()

        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale


    def get_terrain_height(self,p):
        """ Samples heights of the terrain at the required global coordinates point.

        Args:
            p (Tensor): Tensor of shape (n, 2) containing the global xy coordinates of the points to sample

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        points = p.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()

        px = points[:, 0].view(-1)
        py = points[:, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(p.shape[0], -1) * self.terrain.cfg.vertical_scale

    #------------ reward functions----------------

    def _reward_task_pos(self):
        # Reward for completing the task
        
        env_ids = self.episode_length_buf == self.max_episode_length
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        # Base position relative to initial states:
        rel_root_states = self.landing_poses[:,:2] - self.initial_root_states[:,:2]

        tracking_error = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        tracking_error = torch.linalg.norm(rel_root_states[:] - self.commands[:, :2],dim=1)
        # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
        has_jumped_idx = torch.logical_and(self.has_jumped,~self._has_jumped_rand_envs)

        max_tracking_error = self.cfg.env.reset_landing_error #(self.cfg.env.reset_landing_error * (self.commands[:,:2])).clip(min=0.1)

        self.reset_idx_landing_error[torch.logical_and(has_jumped_idx,tracking_error>max_tracking_error)] = True
        

        self.tracking_error_store[has_jumped_idx] = tracking_error[has_jumped_idx]
        self.tracking_error_percentage_store[has_jumped_idx] = tracking_error[has_jumped_idx]/torch.linalg.norm(self.commands[has_jumped_idx,:2],dim=-1)


        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            pass
        else:
            
            # Only give a reward for robots that have landed and are at the end of the episode:
            idx = torch.logical_and(env_ids,has_jumped_idx)

            rew[idx] = torch.exp(-torch.square(tracking_error[idx])/self.cfg.rewards.command_pos_tracking_sigma)


        return rew

    def _reward_task_ori(self):
        # Reward for completing the task
        env_ids = self.episode_length_buf == self.max_episode_length

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        quat_landing = self.landing_poses[:, 3:7]
        quat_des = self.commands[:, 3:7]

        ori_tracking_error = quat_distance(quat_landing, quat_des)

        _,_,yaw_landing = get_euler_xyz(self.landing_poses[:, 3:7])
        _,_,yaw_des = get_euler_xyz(self.commands[:, 3:7])

        ori_tracking_error_yaw = torch.abs(wrap_to_pi(yaw_landing-yaw_des))

        # Check which envs have actually jumped (and not just been initialised at an already "jumped" state)
        has_jumped_idx = torch.logical_and(self.has_jumped,~self._has_jumped_rand_envs)
        self.reset_idx_landing_error[torch.logical_and(has_jumped_idx,ori_tracking_error_yaw>0.5)] = True

        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            pass
        else:
            # Only give a reward for robots that have landed and are at the end of the episode:
            idx = env_ids * has_jumped_idx
            
            rew[idx] = torch.exp(-torch.square(ori_tracking_error_yaw[idx])/self.cfg.rewards.command_ori_tracking_sigma)


        return rew
    
    def _reward_post_landing_pos(self):
        # Reward for remaining at the same position after landing:
        env_ids = self.has_jumped
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            pass
        else:
            
            tracking_error  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
            
            # Track landing position deviation:
            tracking_error[env_ids] = torch.linalg.norm(self.root_states[env_ids,:2] - self.landing_poses[env_ids,:2],dim=1)
            
            # For those that started as has_jumped, track initial position deviation:
            idx = torch.logical_and(env_ids,self._has_jumped_rand_envs)
            tracking_error[idx] = torch.linalg.norm(self.root_states[idx,:2] - self.initial_root_states[idx,:2],dim=1)
            

            rew[env_ids] = torch.exp(-torch.square(tracking_error[env_ids])/self.cfg.rewards.post_landing_pos_tracking_sigma)


        return rew
    
    def _reward_post_landing_ori(self):
        # Reward for remaining at the same orientation after landing:
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        env_ids = self.has_jumped

        quat_ini = self.root_states[:, 3:7]
        quat_des = self.commands[:, 3:7]


        ori_tracking_error = quat_distance(quat_ini, quat_des)
  
        rew[env_ids] = torch.exp(-torch.square(ori_tracking_error[env_ids])/self.cfg.rewards.command_ori_tracking_sigma)


        return rew


    def _reward_jumping(self):
        # Reward if the robot has jumped in the episode:
        env_ids = torch.logical_or(self.episode_length_buf == self.max_episode_length,
                  torch.logical_and(self.reset_buf, self.episode_length_buf < self.max_episode_length))


        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        
        rew[env_ids * self.has_jumped * self.max_height>0.50] = 1        
        
        return rew
     
    
    def _reward_task_max_height(self):
        # Reward for max height achieved during the episode:
        env_ids = torch.logical_and(self.episode_length_buf == self.max_episode_length,self.has_jumped)

        rew  = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        if torch.all(env_ids == False): # if no env is done return 0 reward for all
            return rew
    

        max_height_reward = (self.max_height[env_ids] - 0.9)

        rew[env_ids] = torch.exp(-torch.square(max_height_reward)/self.cfg.rewards.max_height_reward_sigma)


        return rew

    def _reward_change_of_contact(self):
        # Penalty for changing contact state:
        
        rew = torch.sum(torch.abs(self.contacts.int() - self.last_contacts.int()),dim=1)

        rew = torch.exp(-torch.square(rew)/4)

        rew[self.has_jumped * ~self._has_jumped_rand_envs] *= 0.5

        return rew

    def _reward_early_contact(self):
        # Reward maintaining contact at the very beginning of the episode:
        

        env_ids = torch.logical_or((self.episode_length_buf - self.settled_after_init_timer <= 10) * \
                                   (self.episode_length_buf - self.settled_after_init_timer >= 0) * self.settled_after_init,
                                    (self.episode_length_buf - self._has_jumped_switched_time <= 10) *\
                                    (self.episode_length_buf - self._has_jumped_switched_time >= 0) * self.settled_after_init)


        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)
        # Give a reward of 1 if all feet are in contact:
        idx = torch.all(self.contacts,dim=1)
        rew[torch.logical_and(env_ids,idx)] = 1.
        # Give a smaller reward if all feet are in contact when landed:
        rew[self.has_jumped * self.was_in_flight * idx] = 0.2

        return rew
    
    def _reward_feet_distance(self):
        # Reward small feet distance from body

        feet_relative = self.feet_pos[:, :, :3] - self.root_states[:, :3].unsqueeze(1)
        feet_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device, requires_grad=False)
        for i in range(4):
            feet_body_frame[:,i,:] = quat_rotate_inverse(self.base_quat, feet_relative[:,i,:])
        
                
        feet_pos_ini = torch.tensor(self.cfg.init_state.rel_foot_pos).to(self.device).transpose(1,0).view(1,4,3)
        feet_pos_des = feet_pos_ini.clone()

        # In mid-air and above 0.45m height, track close to body (otheriwse track normal):
        feet_pos_des[:,:,2 ]= -0.15

        feet_error = torch.linalg.norm(feet_body_frame - feet_pos_des,dim=-1)
        
        
        rew = torch.sum(torch.square(feet_error),dim=-1)

        # Only reward if in mid_air, hasn't jumped and height is above 0.45
        base_height = self.root_states[:,2] - self.get_terrain_height(self.root_states[:,:2]).flatten()
        rew[base_height<=0.45] = 0.0
        rew[~self.mid_air] = 0.0
        rew[self.has_jumped] = 0.0

        return rew
    

    def _reward_base_height_flight(self):
        # Reward flight height
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)


        if self.jump_type == "upwards":
            base_height_flight = (self.root_states[self.mid_air, 2] - 0.7)
        else:
            base_height_flight = (self.root_states[self.mid_air, 2] - 0.8)

        rew[self.mid_air] = torch.exp(-torch.square(base_height_flight)/self.cfg.rewards.flight_reward_sigma)

        rew[self.has_jumped + ~self.mid_air] = 0.



        return rew 
    
    def _reward_base_height_stance(self):
        # Reward feet height
        base_height = self.root_states[:, 2]
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)


        # Get the height of the terrain at the base position: (to offset the global base height):
        heights = self.get_terrain_height(self.root_states[:,:2]).flatten()

        base_height_stance = (base_height - heights - 0.32)[self.has_jumped]

        squat_idx = torch.logical_and(~self.mid_air,~self.has_jumped)
        base_height_squat = (self.root_states[squat_idx, 2] - 0.20)

        rew[squat_idx] = 0.6*torch.exp(-torch.square(base_height_squat)/self.cfg.rewards.squat_reward_sigma)
        rew[self.has_jumped] =  torch.exp(-torch.square(base_height_stance)/self.cfg.rewards.stance_reward_sigma)
        
        return rew 
    
    def _reward_symmetric_joints(self):
        # Reward the joint angles to be symmetric on each side of the body:
        dof = self.dof_pos.clone().view(self.num_envs, 4, int(self.num_dof/4))
        # # Multiply the right side hips by -1 to match the sign of the left side:
        dof[:,1,0] *= -1
        dof[:,3,0] *= -1
        
        err = torch.sum(torch.abs(dof[:,0,:] - dof[:,1,:]),axis=1) + torch.sum(torch.abs(dof[:,2,:] - dof[:,3,:]),axis=1)
        # Also symmetry on the foot contacts:
        # contacts = self.contacts.float()
        # err += 5*( (torch.abs(contacts[:,0] - contacts[:,1])) + (torch.abs(contacts[:,2] - contacts[:,3])) )

        return err

    
    def _reward_default_pose(self):
        # Penalise large actions:

        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        angle_diff = torch.square(self.dof_pos - self.default_dof_pos)
        angle_diff[self.mid_air + self.has_jumped,0::3] *= 10 # For hips
        

        rew = torch.exp(-torch.sum(angle_diff,dim=1)/self.cfg.rewards.dof_pos_sigma)
        rew[self.has_jumped] *= 2.0


        return rew
    

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_dof_jerk(self):
        return torch.sum(torch.square((self.last_dof_acc - self.dof_acc) / self.dt), dim=1)
    
    def _reward_base_acc(self):
        # Penalize base accelerations
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        base_acc_approx = (self.root_states[:,7:10] - self.last_root_vel[:,:3]) / self.dt
        base_acc_imu = self.imu_state.clone()

        base_acc = base_acc_approx.clone()

        idx_time = torch.logical_or((self.episode_length_buf - self.settled_after_init_timer <= 10) * \
                                    (self.episode_length_buf - self.settled_after_init_timer >= 0) * \
                                    self.settled_after_init,
                                    (self.episode_length_buf - self._has_jumped_switched_time <= 10) * \
                                    (self.episode_length_buf - self._has_jumped_switched_time >= 0) * \
                                    self.settled_after_init)

        rew = torch.square(base_acc)
        rew[idx_time] *= 1e3
        

        return torch.sum(rew, dim=1)
    
    def _reward_energy_usage_actuators(self):
        # Penalize energy usage
        rew = torch.zeros(self.num_envs, device=self.device, requires_grad=False)

        rew = torch.abs(torch.bmm((self.torques).reshape(self.num_envs,1,self.num_dof),(self.dof_vel).reshape(self.num_envs,self.num_dof,1)).flatten())


        return rew


    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
    
    def _reward_action_rate_second_order(self):
        # Penalize changes in action rate

        return torch.sum(torch.square(self.actions - 2*self.last_actions + self.last_last_actions), dim=1)


    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.linalg.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)

        rew = torch.zeros(self.num_envs, device=self.device)
        lin_vel_error = torch.zeros(self.num_envs, device=self.device)
        # Linear velocity commands for flight phase:
        flight_idx = self.mid_air * ~self.has_jumped
        lin_vel_error[flight_idx] = torch.sum(torch.square(self.root_states[flight_idx, 7:9] - self.command_vels[flight_idx, :2]), dim=-1)
        # If told to stand in place, penalise the velocity:
        stance_idx = self.has_jumped * self._has_jumped_rand_envs
        lin_vel_error[stance_idx] = torch.sum(torch.square(self.root_states[stance_idx, 7:9]), dim=-1)
        
        rew = torch.exp(-lin_vel_error/self.cfg.rewards.vel_tracking_sigma)
        rew[~self.has_jumped * ~self.mid_air] = 0
        rew[self.has_jumped * ~self._has_jumped_rand_envs] = 0

        return rew
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw only)
        rew = torch.zeros(self.num_envs, device=self.device)
        ang_vel_error = torch.zeros(self.num_envs, device=self.device) 

        flight_idx = self.mid_air * ~self.has_jumped
        ang_vel_error[flight_idx] = torch.square(self.root_states[flight_idx, 12] - self.command_vels[flight_idx, 5])

        stance_idx = self.has_jumped *  self._has_jumped_rand_envs
        ang_vel_error[stance_idx] = torch.sum(torch.square(self.root_states[stance_idx, 10:13]),dim=-1)
        
        
        rew = torch.exp(-ang_vel_error/self.cfg.rewards.vel_tracking_sigma)

        rew[~self.has_jumped * ~self.mid_air] = 0
        rew[self.has_jumped * ~self._has_jumped_rand_envs] = 0
        
        return rew
   
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)
    
