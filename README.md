# RL-Lite3

## Introduction
A Learning-based locomotion controller for quadruped robots. It includes all components needed for training and hardware deployment on DeepRobotics Lite3.
## Software architecture
This repository consists of below directories:
- rsl_rl: a package wrapping RL methods.
- legged_gym: gym-style environments of quadruped robots.


## Prepare environment 
1.  Create a python (3.6/3.7/3.8, 3.8 recommended) environment on Ubuntu OS.

2.  Install pytorch with cuda.
```
# pytorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

3.  Download Isaac Gym (version >=preview 3) from the official website and put it into the root directory of the project.

4. Install python dependencies with pip.
```
pip3 install transformations matplotlib gym tensorboard numpy=1.23.5
```

5. Install legged_gym and rsl_rl by pip
```
cd legged_gym
pip install -e .

cd rsl_rl
pip install -e .
```

# Usage

### Train policy in the simulation
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/train.py --rl_device cuda:0 --sim_device cuda:0 --headless
```

### Run controller in the simulation
```
cd ${PROJECT_DIR}
python3 legged_gym/legged_gym/scripts/play.py --rl_device cuda:0 --sim_device cuda:0 --load_run ${model_dir} --checkpoint ${model_name}
```
Check that your computer has a GPU, otherwise, replace the word `cuda:0` with `cpu`.
You should assign the path of the network model via `--load_run` and `--checkpoint`. 

### Run controller in the real-world

Copy your policy file to the project [rl_deploy(https://github.com/DeepRoboticsLab/deeprobotics_rl_deploy.git)],then you can run your reinforcement learning controller in the real world

## Reference
[legged_gym] https://github.com/leggedrobotics/legged_gym.git
[rsl_rl]https://github.com/leggedrobotics/rsl_rl
[quadruped-robot]https://gitee.com/HUAWEI-ASCEND/quadruped-robot.git




## Communication
https://www.deeprobotics.cn/robot/index/company.html#maps
