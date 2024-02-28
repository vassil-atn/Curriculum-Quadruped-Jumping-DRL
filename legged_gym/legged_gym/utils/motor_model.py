import numpy as np
from scipy.signal import butter
import matplotlib.pyplot as plt
import time
import torch

class MotorModel():

    def __init__(self, device):
        self.device = device
        self.Kt = 0.63895 # Nm/A