conda create -n RLjumping python=3.8
conda activate RLjumping
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:Vassil17/Curriculum-Quadruped-Jumping-DRL.git
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
# Trained with Preview 4
cd isaacgym/python && pip install -e .
cd ~/Curriculum-Quadruped-Jumping-DRL/rsl_rl && pip install -e .
cd ~/Curriculum-Quadruped-Jumping-DRL/legged_gym && pip install -e .