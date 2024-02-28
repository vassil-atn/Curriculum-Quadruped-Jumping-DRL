import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import time
import torch

class ButterworthFilter():

    def __init__(self, order, f_cutoff,sampling_rate, q0, num_envs, num_joints, device, ftype='low'):
        self.device = device
        self.a = torch.zeros(num_envs,num_joints,order+1,device=self.device)
        self.b = torch.zeros(num_envs,num_joints,order+1,device=self.device)

        nyq = 0.5*sampling_rate
        fc = f_cutoff / nyq
        b,a = butter(order,[fc],btype=ftype)

        self.a[:,:,:] = torch.from_numpy(a)
        self.b[:,:,:] = torch.from_numpy(b)

        self.a = self.a / a[0]

        self.b = self.b / a[0]

        self.q0 = q0

        self.xhist = torch.zeros(num_envs,num_joints,order,device=self.device)
        self.yhist = torch.zeros(num_envs,num_joints,order,device=self.device)

    def reset(self,env_ids,control_type):
        if control_type == 'P_joint_pos':
            self.xhist[env_ids,:,:] = self.q0
            self.yhist[env_ids,:,:] = self.q0
        else:
            self.xhist[env_ids,:,:] = 0.
            self.yhist[env_ids,:,:] = 0.

    def filter(self,x):
        xs = self.xhist
        ys = self.yhist

        y = torch.multiply(x, self.b[:, :, 0]) + torch.sum(
            torch.multiply(xs, self.b[:, :, 1:]), dim=-1) - torch.sum(
                torch.multiply(ys, self.a[:, :, 1:]), dim=-1)
        
        self.xhist = torch.roll(self.xhist, 1, dims=-1)
        self.xhist[:,:,0] = x
        self.yhist = torch.roll(self.yhist, 1, dims=-1)
        self.yhist[:,:,0] = y

        return y


    def init_history(self, x):
        self.xhist[:,:] = x
        self.yhist[:,:] = x



