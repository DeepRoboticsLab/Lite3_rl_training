import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)
import numpy as np
import json
from datetime import datetime
import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, Logger, register
from legged_gym.utils.task_registry import task_registry
from legged_gym.utils.helpers import class_to_dict


def train(args):
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for training
    env_cfg.commands.fixed_commands = None

    # prepare environment
    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load model
    if args.load_run:
        train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg,
        enable_summary_writer=True)

    # record configs as log files
    if not os.path.exists(ppo_runner.log_dir):
        os.mkdir(ppo_runner.log_dir)
    with open(os.path.join(ppo_runner.log_dir, 'env_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(env_cfg), fp)
    with open(os.path.join(ppo_runner.log_dir, 'train_cfg.json'), 'w') as fp:
        json.dump(class_to_dict(train_cfg), fp)

    # train ppo policy
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == '__main__':
    args = get_args()
    args.save_rewards = True
    train(args)
