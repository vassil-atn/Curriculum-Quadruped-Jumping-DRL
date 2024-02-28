# Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design
by Vassil Atanassov, Jiatao Ding, Jens Kober, Ioannis Havoutis, Cosimo Della Santina


If you use this repository in your work, consider citing:
```
@misc{atanassov2024curriculumbased,
      title={Curriculum-Based Reinforcement Learning for Quadrupedal Jumping: A Reference-free Design}, 
      author={Vassil Atanassov and Jiatao Ding and Jens Kober and Ioannis Havoutis and Cosimo Della Santina},
      year={2024},
      eprint={2401.16337},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
## Video
https://www.youtube.com/watch?v=nRaMCrwU5X8
[![](https://img.youtube.com/vi/nRaMCrwU5X8/0.jpg)](https://www.youtube.com/watch?v=nRaMCrwU5X8)

## Installation
```bash
conda create -n RLjumping python=3.8
conda activate RLjumping
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:Vassil17/Curriculum-Quadruped-Jumping-DRL.git
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
# Trained with Preview 4
cd isaacgym/python && pip install -e .
cd ~/Curriculum-Quadruped-Jumping-DRL/rsl_rl && pip install -e .
cd ~/Curriculum-Quadruped-Jumping-DRL/legged_gym && pip install -e .
cd ~/Curriculum-Quadruped-Jumping-DRL && pip install -r requirements.txt
```

## Training
1. To train an upwards jumping policy simply run:
```
cd ~/Curriculum-Quadruped-Jumping-DRL/legged_gym/scripts
python train.py --task="go1_upwards" --max_iterations=3000 --headless
```
2. Then for the second curriculum stage, do:
```
python train.py --task="go1_forward" --max_iterations=10000 --headless --resume --load_run=RUN_ID
```
where RUN_ID is the training run ID you want to resume from (leave -1 for the last one.)

To evaluate your policy you can play around with the settings in `scripts/test.py`.

Disclaimers: Evaluating highly dynamic jumps on the real hardware can be dangerous, so carefully test policies in the simulation first. 
Unfortunately, due to a reported issue with Isaac Gym operating on trimesh being non-deterministic you might observe variations between training runs (despite the same random seed). 

## Acknowledgments
Code-base is based on the following works:

https://github.com/leggedrobotics/legged_gym

https://github.com/Improbable-AI/walk-these-ways

