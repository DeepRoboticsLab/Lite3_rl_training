import os
currentdir = os.path.dirname(os.path.abspath(__file__))
legged_gym_dir = os.path.dirname(os.path.dirname(currentdir))
isaacgym_dir = os.path.join(os.path.dirname(legged_gym_dir), "isaacgym/python")
rsl_rl_dir = os.path.join(os.path.dirname(legged_gym_dir), "rsl_rl")
os.sys.path.insert(0, legged_gym_dir)
os.sys.path.insert(0, isaacgym_dir)
os.sys.path.insert(0, rsl_rl_dir)

import numpy as np
import csv
from legged_gym.utils import get_args, Logger, register


def play(args):
    from legged_gym.utils.task_registry import task_registry
    from legged_gym import LEGGED_GYM_ROOT_DIR
    import torch
    record_policy_output = False
    register(args.task, task_registry)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = 10

    env_cfg.viewer.real_time_step = True
    env_cfg.pmtg.train_mode = False
    # env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    # env_cfg.terrain.evaluation_mode = True

    # customized terrain mode
    env_cfg.terrain.selected = True
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.commands.fixed_commands = [0.8, 0.0, 0.0]
    # env_cfg.viewer.debug_viz = True
    env_cfg.terrain.terrain_length = 8
    env_cfg.terrain.terrain_width = 8
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 2
    env_cfg.env.episode_length_s = 100
    env_cfg.terrain.slope_treshold = 0.5  # for stair generation
    # env_cfg.terrain.terrain_kwargs = {'type': 'sloped_terrain', 'slope': 0.26}
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5}]
    # env_cfg.terrain.terrain_kwargs = [{'type': 'slope_platform_stairs_terrain', 'slope': 0.36, 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5},
    #                                   {'type': 'stairs_platform_slope_terrain', 'step_width': 0.2, 'step_height': 0.1, 'num_steps': 5, 'slope': 0.36}]
    env_cfg.terrain.terrain_kwargs = [{
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': -0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_stairs_terrain',
        'step_width': 0.3,
        'step_height': 0.1,
        'platform_size': 3.
    }, {
        'type': 'pyramid_sloped_terrain',
        'slope': 0.26
    }, {
        'type': 'discrete_obstacles_terrain',
        'max_height': 0.10,
        'min_size': 0.1,
        'max_size': 0.5,
        'num_rects': 200
    }, {
        'type': 'wave_terrain',
        'num_waves': 4,
        'amplitude': 0.15
    }, {
        'type': 'stepping_stones_terrain',
        'stone_size': 0.1,
        'stone_distance': 0.,
        'max_height': 0.03
    }]
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)

    obs_dict = env.get_observations()
    obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]    
    if record_policy_output:
        csv_header = [str(i) for i in range(env.num_policy_outputs)]
        with open(os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, args.load_run,
                                'policy_outputs.csv'),
                    'w',
                    newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)
    for i in range(10 * int(env.max_episode_length)):
        with torch.no_grad():
            actions = policy(obs, obs_history)
            # print(actions[0])
        obs_dict, rews, dones, infos = env.step(actions)
        obs, privileged_obs, obs_history = obs_dict["obs"], obs_dict["privileged_obs"], obs_dict["obs_history"]


if __name__ == '__main__':
    args = get_args()
    play(args)
