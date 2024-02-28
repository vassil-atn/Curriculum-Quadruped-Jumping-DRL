import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import time
import torch


class EMAFilter():

    def __init__(self, f_cutoff,sampling_rate, q0, num_envs, num_joints, device):
        self.f_cutoff = f_cutoff
        self.fs = sampling_rate
        self.device = device
        self.num_envs = num_envs
        self.q0 = q0
        self.num_joints = num_joints

        self.filtered_values = torch.zeros(self.num_envs, self.num_joints, dtype=torch.float, device=self.device, requires_grad=False)

        w = 2*np.pi*self.f_cutoff / self.fs
        self.alpha = np.cos(w) - 1 + np.sqrt((np.cos(w))**2 - 4*np.cos(w) + 3) 
        print(f"EMA filter alpha: {self.alpha}")


    def reset(self,env_ids,control_type):
        if control_type == 'P_joint_pos':
            self.filtered_values[env_ids,:] = self.q0
        else:
            self.filtered_values[env_ids,:] = 0.0

    def filter(self,actions):

        filtered_actions = (1-self.alpha)*self.filtered_values + self.alpha*actions
        self.filtered_values = filtered_actions.clone()

        return filtered_actions
    
