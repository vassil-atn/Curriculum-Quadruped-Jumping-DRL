
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np
import torch
import time
from scipy.interpolate import interp1d
from isaacgym.torch_utils import *
from legged_gym.utils.math import quat_distance,quat_slerp,quat_exp,torch_rand_float_tensor,wrap_to_pi

import matplotlib.pyplot as plt
from matplotlib import colors
# import roma
import random
from scipy.spatial.transform import Rotation as R
# from scipy.spatial.transform import Slerp

def test(args):
    torch.manual_seed(0)
    np.random.seed(0)
    args.task = "go1_forward"
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)

    # Terrain:
    env_cfg.commands.randomize_commands = True

    # env_cfg.terrain.curriculum = True
    # env_cfg.terrain.terrain_kwargs = { 'type':'box_terrain', 'box_width': 0.6, 'box_height':0.6}
    env_cfg.terrain.terrain_length = 2.
    env_cfg.terrain.terrain_width = 2.
    env_cfg.terrain.border_size = 2.
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 6
    env_cfg.terrain.sloped_terrain_number = 2
    env_cfg.terrain.num_zero_height_terrains = 0
    env_cfg.terrain.measure_heights = False
    env_cfg.terrain.make_terrain_uneven = False

    #
#----------------------Noise and domain rand:--------------------------------
    env_cfg.noise.add_noise = False
    env_cfg.noise.noise_level = 1.
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.ranges.friction_range = [1.0,1.0]
    
    env_cfg.domain_rand.randomize_robot_pos = False
    env_cfg.domain_rand.randomize_robot_vel = False
    env_cfg.domain_rand.push_upwards = False
    # env_cfg.domain_rand.ranges.min_robot_vel = [-0.0, -0.0,4.5]
    # env_cfg.domain_rand.ranges.max_robot_vel = [0.,0.,4.5]
    env_cfg.domain_rand.randomize_robot_ori = False
    # # env_cfg.domain_rand.pos_vel_random_prob = 1.
    env_cfg.domain_rand.randomize_dof_pos = False
    env_cfg.domain_rand.randomize_spring_params = False
    env_cfg.domain_rand.randomize_motor_strength = False
    env_cfg.domain_rand.randomize_PD_gains = False
    env_cfg.domain_rand.randomize_motor_offset = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.ranges.added_mass_range = [4.,4.]
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.ranges.added_link_mass_range = [1.3,1.3]
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.ranges.joint_armature_range = [1e-2,1e-2]

    
    
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.domain_rand.ranges.restitution_range = [0.0,0.0]
    env_cfg.domain_rand.randomize_com = False
    env_cfg.domain_rand.ranges.com_displacement_range = [0.1,0.1]
#-------------------------------------------------------
    env_cfg.domain_rand.randomize_has_jumped = True
    env_cfg.domain_rand.has_jumped_random_prob = 1.0
    env_cfg.domain_rand.reset_has_jumped = True
    env_cfg.domain_rand.manual_has_jumped_reset_time = 100
#-------------------------------------------------------
    env_cfg.domain_rand.randomize_joint_friction = True
    env_cfg.domain_rand.randomize_joint_damping = True
    env_cfg.domain_rand.ranges.joint_friction_range = [0.04,0.04]
    env_cfg.domain_rand.ranges.joint_damping_range = [0.01,0.01]

#   CPU VS GPU DIFFERENCE

    env_cfg.domain_rand.push_robots = False
    env_cfg.env.episode_length_s = 5
    # env_cfg.domain_rand.curriculum = False


    env_cfg.domain_rand.push_towards_goal = False
    env_cfg.domain_rand.sim_latency = True
    env_cfg.domain_rand.randomize_lag_timesteps = False

    # env_cfg.domain_rand.sim_pd_latency = False
    env_cfg.domain_rand.ranges.latency_range = [40.0,40.0]
    env_cfg.domain_rand.lag_timesteps = 8
    # env_cfg.domain_rand.ranges.pd_latency_range = [0.0,0.0]
    env_cfg.domain_rand.ranges.additional_latency_range = [-0.0,0.0]
    env_cfg.domain_rand.randomize_gravity = False


    
    # env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.commands.jump_over_box = False

    # Commands
    env_cfg.commands.curriculum = False
    # env_cfg.commands.ranges.pos_dx_ini = [0.9,0.9]
    # env_cfg.commands.ranges.pos_dy_ini = [0.0,0.0]
    # env_cfg.commands.distances.des_yaw = torch.pi
    env_cfg.commands.randomize_yaw = False
    env_cfg.commands.upward_jump_probability = 0.

    env_cfg.env.throttle_to_real_time = False
    env_cfg.viewer.camera_track_robot = False
    env_cfg.viewer.ref_env = 0
    env_cfg.viewer.simulate_camera = False

    # if env_cfg.viewer.simulate_camera:
    # env_cfg.asset.file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go1/urdf/go1_unitree_visual.urdf'
    # env_cfg.asset.flip_visual_attachments = True

    env_cfg.env.debug_draw = True
    env_cfg.env.debug_draw_line_goal = False
    env_cfg.env.continuous_jumping = False
    env_cfg.env.continuous_jumping_reset_probability = 0.0
    # env_cfg.env.use_springs = True

    env_cfg.control.safety_clip_actions = True

    env_cfg.env.reset_height = 0.02

    # prepare environment
    args.load_run = "Aug04_11-10-21_"
    # args.checkpoint = 4000


# 

    # args.headless = True



    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    mean_tracking_error_per_step = []
    joint_angles = []
    joint_vels = []
    joint_acc = []
    joint_jerk = []
    action_rate = []
    action_rate_2 = []
    foot_pos = []
    feet_vel = []
    feet_acc = []
    contact_forces = []
    torques = []
    spring_torques = []
    reward = []
    actions_stored = []
    energy_used = []
    energy_used_act = []
    base_lin_vel = []
    base_ang_vel = []
    contacts = []
    projected_gravity = []
    mid_air = []
    has_jumped = []
    actions_scaled_stored = []
    filtered_actions_stored = []
    dof_acc = []
    base_pos = []
    base_vel_global = []
    base_acc = []
    imu_state = []
    euler = []
    noise = []
    base_lin_vel_imu = []
    des_angles_euler = []
    base_ang_vel_global = []
    ori_error = []

    reset_envs = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    fail_count = 0
    first_time = True
    torque_limits_satisfied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    dof_limits_satisfied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    dof_vel_limits_satisfied = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    dof_vel_limits_greatly_exceeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env_to_store = np.arange(0,env.num_envs)


    env.additional_termination_conditions = False

    for i in range(int(1*env.max_episode_length-3)):

        actions = policy(obs.detach())
        if i == 0:
            actions[:] = 0.0

        action_rate.append((actions.detach().cpu().numpy() - env.last_actions.detach().cpu().numpy())/env.dt)
        action_rate_2.append((actions.detach().cpu().numpy() - 2*env.last_actions.detach().cpu().numpy() + env.last_last_actions.detach().cpu().numpy())/env.dt)
        obs, _, rews, dones, infos = env.step(actions.detach())
        
        if first_time:
            saved_commands = env.commands.clone().cpu().numpy()
            command_vels = env.command_vels.clone().cpu().numpy()

            first_time = False

        torques.append(env.torques[env_to_store].detach().cpu().numpy())
        joint_acc.append(env.dof_acc[env_to_store].cpu().numpy())
        joint_jerk.append(env.dof_jerk[env_to_store].cpu().numpy())
        foot_pos.append(env.feet_pos[env_to_store,:,:].cpu().numpy())
        feet_vel.append(env.feet_vel[env_to_store,:,:].cpu().numpy())
        feet_acc.append(env.feet_acc[env_to_store,:,:].cpu().numpy())
        mid_air.append(env.mid_air[env_to_store].cpu().numpy())
        contact_forces.append(env.contact_forces[:, env.feet_indices, 2].cpu().numpy())
        reward.append(env.rew_buf[env_to_store].detach().cpu().numpy())
        actions_stored.append(actions.detach().cpu().numpy()[env_to_store,:])
        filtered_actions_stored.append(env.actions_filtered[env_to_store,:].detach().cpu().numpy())
        joint_angles.append(env.dof_pos[env_to_store,:].cpu().numpy())
        joint_vels.append(env.dof_vel[env_to_store,:].cpu().numpy())
        energy_used.append(env.torques_to_apply[env_to_store].detach().cpu().numpy())
        energy_used_act.append(env.torques[env_to_store].detach().cpu().numpy()) 
        base_lin_vel.append(env.base_lin_vel[env_to_store].cpu().numpy())
        base_ang_vel.append(env.base_ang_vel[env_to_store].cpu().numpy())

        contacts.append(env.contacts[env_to_store].cpu().numpy())
        projected_gravity.append(env.projected_gravity[env_to_store].cpu().numpy())

        has_jumped.append(env.has_jumped[env_to_store].cpu().numpy())
        spring_torques.append(env.torques_springs[env_to_store].cpu().numpy())
        actions_scaled_stored.append(env.actions_scaled[env_to_store].cpu().numpy())
        dof_acc.append(env.dof_acc[env_to_store].cpu().numpy())
        base_pos.append(env.root_states[env_to_store,0:3].cpu().numpy())
        base_vel_global.append(env.root_states[env_to_store,7:10].cpu().numpy())
        base_acc.append(env.base_acc[env_to_store].cpu().numpy())
        imu_state.append(env.imu_state[env_to_store].cpu().numpy())
        euler.append(env.euler[env_to_store].cpu().numpy())
        base_lin_vel_imu.append(env.base_lin_vel_imu[env_to_store].cpu().numpy())
        des_angles_euler.append(env.des_angles_euler[env_to_store].cpu().numpy())
        base_ang_vel_global.append(env.root_states[env_to_store,10:13].cpu().numpy())
        ori_error.append(env.ori_error[env_to_store].cpu().numpy())

        max_height = env.max_height[env_to_store].cpu().numpy()
        

        reset_envs[env.reset_buf * ~env.time_out_buf] = True
        fail_count += torch.count_nonzero(env.reset_buf * ~env.time_out_buf)
        dof_limits_satisfied[torch.logical_or(torch.any((env.dof_pos - env.dof_pos_limits_urdf[:,0]) < -0.1,dim=-1), 
                                              torch.any((env.dof_pos - env.dof_pos_limits_urdf[:,1]) > 0.1, dim=-1))] = False
        dof_vel_limits_satisfied[torch.any((torch.abs(env.dof_vel) - env.dof_vel_limits) > 5.0, dim=-1)] = False
        torque_limits_satisfied[torch.any((torch.abs(env.torques) - env.torque_limits) > 2.0, dim=-1)] = False
        dof_vel_limits_greatly_exceeded[torch.logical_and(~env.was_in_flight,torch.any((torch.abs(env.dof_vel) - env.dof_vel_limits) > 10.0, dim=-1))] = True


        if torch.any(env.has_jumped * env.was_in_flight):
            print('Max: ',env.max_height[env_to_store])
            print('Min: ',env.min_height[env_to_store])
      
        mean_tracking_error_per_step.append(torch.mean(env.tracking_error_store[env.tracking_error_store!=0.0]).to('cpu'))

     
    mid_air_tensor = torch.tensor(np.array(mid_air))
    flight_time_all = torch.count_nonzero(mid_air_tensor,dim=0) * env.dt
    flight_time_mean = torch.mean(flight_time_all)

    print(f"Mean dof acc {env.mean_dof_acc_stored.cpu()}")
    # print(f"Max dof acc {env.mean_dof_acc_stored.cpu()}")
    print(f"Mean base acc {env.mean_base_acc_stored.cpu()}")
    # print(f"Max base acc {torch.max(torch.abs(env.base_acc_stored.cpu()))}")
    print(f"Mean action rate {env.mean_action_rate_stored.cpu()}")
    # print(f"Max action rate {torch.max(torch.abs(env.action_rate_stored.cpu()))}")
    print(f"Mean action rate 2 {env.mean_action_rate_2_stored.cpu()}")
    # print(f"Max action rate 2 {torch.max(torch.abs(env.action_rate_2_stored.cpu()))}")


    print(f"Mean tracking error as %: {torch.mean(env.tracking_error_percentage_store[env.tracking_error_percentage_store!=0.0])}")
    print(f"Torque limits satisfied: {torch.all(torque_limits_satisfied)}")
    print(f"Joint pos limits satisfied: {torch.all(dof_limits_satisfied)}")
    print(f"Joint vel limits satisfied: {torch.all(dof_vel_limits_satisfied)}")
    print(f"Environments where dof vel limits failed: {torch.nonzero(~dof_vel_limits_satisfied)}")
    print(f"Exceeded dof vel over 10? {torch.any(dof_vel_limits_greatly_exceeded)}")
    print(f"Failed jumps (original set): {torch.count_nonzero(reset_envs)}")
    print(f"Failed jumps (total): {fail_count}")
    des_env =1#torch.nonzero(~dof_vel_limits_satisfied).cpu().numpy().flatten()

    joint_angles = np.array(joint_angles).swapaxes(0,1)[des_env]
    joint_vels = np.array(joint_vels).swapaxes(0,1)[des_env]
    action_rate = np.array(action_rate).swapaxes(0,1)[des_env]
    action_rate_2 = np.array(action_rate_2).swapaxes(0,1)[des_env]
    torques = np.array(torques).swapaxes(0,1)[des_env] 
    joint_acc = np.array(joint_acc).swapaxes(0,1)[des_env] 
    joint_jerk = np.array(joint_jerk).swapaxes(0,1)[des_env] 
    foot_pos = np.array(foot_pos).swapaxes(0,1)[des_env] 
    feet_vel = np.array(feet_vel).swapaxes(0,1)[des_env]
    feet_acc = np.array(feet_acc).swapaxes(0,1)[des_env]
    contact_forces = np.array(contact_forces).swapaxes(0,1)[des_env]
    reward = np.array(reward).swapaxes(0,1)[des_env] 
    actions_stored = np.array(actions_stored).swapaxes(0,1)[des_env] 
    filtered_actions_stored = np.array(filtered_actions_stored).swapaxes(0,1)[des_env]
    energy_used = np.array(energy_used).swapaxes(0,1)[des_env] 
    energy_used_act = np.array(energy_used_act).swapaxes(0,1)[des_env] 
    base_lin_vel = np.array(base_lin_vel).swapaxes(0,1)[des_env]
    base_ang_vel = np.array(base_ang_vel).swapaxes(0,1)[des_env]
    contacts = np.array(contacts).swapaxes(0,1)[des_env]
    has_jumped = np.array(has_jumped).swapaxes(0,1)[des_env]
    # error_quat = np.array(error_quat).swapaxes(0,1)[des_env]
    projected_gravity = np.array(projected_gravity).swapaxes(0,1)[des_env]
    spring_torques = np.array(spring_torques).swapaxes(0,1)[des_env]
    actions_scaled_stored = np.array(actions_scaled_stored).swapaxes(0,1)[des_env]
    dof_acc = np.array(dof_acc).swapaxes(0,1)[des_env]
    base_pos = np.array(base_pos).swapaxes(0,1)[des_env]
    base_vel_global = np.array(base_vel_global).swapaxes(0,1)[des_env]
    base_acc = np.array(base_acc).swapaxes(0,1)[des_env]
    imu_state = np.array(imu_state).swapaxes(0,1)[des_env]
    euler = np.array(euler).swapaxes(0,1)[des_env]
    base_lin_vel_imu = np.array(base_lin_vel_imu).swapaxes(0,1)[des_env]
    des_angles_euler = np.array(des_angles_euler).swapaxes(0,1)[des_env]
    base_ang_vel_global = np.array(base_ang_vel_global).swapaxes(0,1)[des_env]
    mid_air = np.array(mid_air).swapaxes(0,1)[des_env]
    ori_error = np.array(ori_error).swapaxes(0,1)[des_env]
    command_vels = command_vels[des_env]


    print(f"Mean flight time: {flight_time_mean}")
    reset_envs = reset_envs.cpu().numpy()
    col = np.where(reset_envs==True,0,np.where(reset_envs==False,1,2))
    cmap = colors.ListedColormap(['r','k','g'])

    # Convert to array of integers with range 1 to 40 depending on percentage error
    percentage_error = env.tracking_error_percentage_store.cpu().numpy() * 100
    # m = interp1d([0,np.max(percentage_error)],[20,100])

    s = (percentage_error+10).astype(int)

    plt.rcParams.update({'font.size': 24})
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['figure.figsize'] = (20, 10)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['legend.framealpha'] = 1.

    fig,ax = plt.subplots(1)
    fig.suptitle('Success rate')
    if env_cfg.env.jump_type == "forward":
        scatter = ax.scatter(saved_commands[:,0], saved_commands[:,1], c=col, cmap=cmap,s=s, marker='X')
    else:
        pass

    landing_poses = env.landing_poses.cpu().numpy()
    initial_root_states = env.initial_root_states.cpu().numpy()
 

    fig,ax = plt.subplots(1)
    fig.suptitle('Desired Distance vs Actual Distance')
    landing_poses_filtered = landing_poses.copy()
    idx = np.isnan(landing_poses_filtered[:,2])
    landing_poses_filtered[idx,2] = 0.0
    achieved_dist = np.linalg.norm(landing_poses[:,:2] - initial_root_states[:,:2],axis=-1)# - saved_commands[:,:2],axis=-1)
    dist = np.linalg.norm(saved_commands[:,:2],axis=-1)
    ax.scatter(dist,achieved_dist, c=col, s=20, marker='X')
    line = np.linspace(0,ax.get_xlim(),100)
    ax.plot(line,line,'k--')
    ax.set_xlabel('Desired Distance [m]')
    ax.set_ylabel('Actual Distance [m]')



    des_joint_angles = env_cfg.init_state.spring_rest_pos.cpu().numpy() + actions_scaled_stored

    fig,ax = plt.subplots(4,1)
    fig.suptitle('Joint angles')
    colours = ['b','g','r']
    for i in range(4):
        ax[i].plot(joint_angles[:,3*i+0],'b')
        ax[i].plot(joint_angles[:,3*i+1],'g')
        ax[i].plot(joint_angles[:,3*i+2],'r')
        ax[i].plot(des_joint_angles[:,3*i+0],'b--')
        ax[i].plot(des_joint_angles[:,3*i+1],'g--')
        ax[i].plot(des_joint_angles[:,3*i+2],'r--')
        for j in range(3):
            ax[i].axhline(y=env.dof_pos_limits_urdf.cpu().numpy()[i*3+j,0], color=colours[j], linestyle='--')
            ax[i].axhline(y=env.dof_pos_limits_urdf.cpu().numpy()[i*3+j,1], color=colours[j], linestyle='--')
    fig,ax = plt.subplots(4,1)
    fig.suptitle('Joint velocities')
    for i in range(4):
        ax[i].plot(joint_vels[:,3*i+0],'b')
        ax[i].plot(joint_vels[:,3*i+1],'g')
        ax[i].plot(joint_vels[:,3*i+2],'r')
        for j in range(3):
            ax[i].axhline(y=env.dof_vel_limits.cpu().numpy()[i*3+j], color=colours[j], linestyle='--')
            ax[i].axhline(y=-env.dof_vel_limits.cpu().numpy()[i*3+j], color=colours[j], linestyle='--')
    fig,ax = plt.subplots(4,3)
    fig.suptitle('Joint Torques')
    ax = ax.ravel()
    for i in range(12):
        ax[i].plot(torques[:,i],'r')
        ax[i].plot(energy_used[:,i],'g--')
        ax[i].plot(spring_torques[:,i],'b')
        ax[i].axhline(y=env.torque_limits.cpu().numpy()[i], color=colours[j], linestyle='--')
        ax[i].axhline(y=-env.torque_limits.cpu().numpy()[i], color=colours[j], linestyle='--')


    fig,ax = plt.subplots(1)
    fig.suptitle('Joint Friction Coeffs vs Flight time')
    ax.scatter(np.mean(env.joint_friction_coeffs.cpu().numpy(),axis=-1),flight_time_all.cpu().numpy())
    fig,ax = plt.subplots(4,3)
    fig.suptitle('Joint Acceleration and Jerk')
    ax = ax.ravel()
    for i in range(12):
        ax[i].plot(dof_acc[:,i],'b')
        ax[i].plot(joint_jerk[:,i],'g--')

    fig,ax = plt.subplots(1,1)
    fig.suptitle('Contact states')
    for i in range(4):
        ax.plot(contacts[:,i],label=f"Foot {i}")
        ax.plot(mid_air[:],'k--')
        ax.legend()

    fig,ax = plt.subplots(4,1)
    fig.suptitle('Base orientation')
    labels = ['Roll','Pitch','Yaw','Ori Error']
    labels_des = ['Des Roll','Des Pitch','Des Yaw','Des Ori Error']
    for i in range(3):
        ax[i].plot(euler[:,i],'b',label=labels[i])
        ax[i].plot(des_angles_euler[:,i],'g',label=labels_des[i])
        ax[i].legend()

    ax[3].plot(ori_error[:],'b',label=labels[3])
    ax[3].plot(0*ori_error[:],'g',label=labels_des[3])
    ax[3].legend()


    fig,ax = plt.subplots(3,3)
    fig.suptitle('Base Position, Velocity, Acceleration')
    labels = ['X','Y','Z']
    command_vels = command_vels[:3] * np.ones_like(base_vel_global)
    command_vels[~mid_air,:] = 0.0
    for i in range(3):
        ax[i,0].plot(base_pos[:,i] - initial_root_states[des_env,i],label=f"{labels[i]} pos")
        ax[i,1].plot(base_vel_global[:,i],label=f"{labels[i]} vel")
        ax[i,1].plot(command_vels[:,i],'g',label=f"{labels[i]} des vel")
        ax[i,1].plot(base_lin_vel_imu[:,i],'r',label=f"{labels[i]} vel imu")
        ax[i,2].plot(base_acc[:,i],label=f"{labels[i]} acc")
        ax[i,2].plot(imu_state[:,i],'g',label=f"IMU {labels[i]}")

    fig,ax = plt.subplots(3,1)
    fig.suptitle('Feet vel')
    for i in range(4):
        ax[0].plot(feet_vel[:,i,0],label=f"Foot {i}, X")
        ax[1].plot(feet_vel[:,i,1],label=f"Foot {i}, Y")
        ax[2].plot(feet_vel[:,i,2],label=f"Foot {i}, Z")

        ax[0].legend()
        ax[1].legend()
        ax[2].legend()
    ax[0].plot(mid_air[:],'k--',label="Mid air")
    ax[1].plot(mid_air[:],'k--',label="Mid air")
    ax[2].plot(mid_air[:],'k--',label="Mid air")
    rew_logs = env.reward_logs
    fig,ax = plt.subplots(8,3)
    ax = ax.ravel()
    fig.suptitle('Rewards')
    i = 0
    k = 0
    for name in rew_logs:

        rew = torch.stack(rew_logs[name]).detach().cpu().numpy()
        ax[i].plot(rew,label=name)
        ax[i].legend(prop = { "size": 12 })
        i += 1

    plt.show()

if __name__ == '__main__':
    args = get_args()
    test(args)
