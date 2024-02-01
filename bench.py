import gymnasium
import numpy as np
import copy
import os
import argparse

import stable_baselines3
from stable_baselines3 import TD3, PPO, A2C, SAC, DQN
from stable_baselines3.common.logger import configure
#from stable_baselines3.common.callbacks import EvalCallback
from my_eval import EvalCallback
from gymnasium.wrappers import TransformReward, PassiveEnvChecker, OrderEnforcing, TimeLimit
from gymnasium.experimental.wrappers import RescaleActionV0
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.monitor import Monitor


def make_env(env_id: str):
    if env_id == "go1":
        env = gymnasium.make(
            'Ant-v5',
            xml_file='../mujoco_menagerie/unitree_go1/scene.xml',
            forward_reward_weight=1,
            ctrl_cost_weight=0.05,
            contact_cost_weight=5e-4,
            healthy_reward=1,
            main_body=1,
            healthy_z_range=(0.195, 0.75),
            include_cfrc_ext_in_observation=False,
            exclude_current_positions_from_observation=False,
            reset_noise_scale=0.1,
            frame_skip=25,
            max_episode_steps=1000,
        )
        return env
    else:
        return gymnasium.make(env_id, render_mode=None)


def make_model(algorithm: str):
    match args.algo:
        case "TD3":  # note does not work with Discrete
            action_noise = NormalActionNoise(mean=np.zeros(12), sigma=0.1 * np.ones(12))
            return TD3("MlpPolicy", env, seed=run, action_noise=action_noise, verbose=1, device='cuda', learning_starts=100, batch_size=256)
        case "PPO":
            return PPO("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "SAC":  # note does not work with Discrete
            action_noise = NormalActionNoise(mean=np.zeros(12), sigma=0.1 * np.ones(12))
            return SAC("MlpPolicy", env, seed=run, action_noise=action_noise, verbose=1, device='cuda', learning_starts=100, batch_size=256)
        case "A2C":
            return A2C("MlpPolicy", env, seed=run, verbose=1, device='cuda')
        case "DQN":
            return DQN("MlpPolicy", env, seed=run, verbose=1, device='cuda', learning_starts=100, batch_size=256)


parser = argparse.ArgumentParser()
parser.add_argument("--algo", default="DQN")
parser.add_argument("--env_id", default="CartPole-v1")
parser.add_argument("--starting_run", default=0, type=int)
args = parser.parse_args()

RUNS = 1  # Number of Statistical Runs
TOTAL_TIME_STEPS = int(10e6)
EVAL_SEED = 1234
EVAL_FREQ = 5000
EVAL_ENVS = 100


for run in range(args.starting_run, RUNS):
    env = Monitor(RescaleActionV0(make_env(args.env_id), min_action=-1, max_action=1))
    eval_env = copy.deepcopy(env)
    eval_path = f"results/{args.env_id}/{args.algo}/run_" + str(run)


    assert not os.path.exists(eval_path)

    #eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True)
    eval_callback = EvalCallback(eval_env, best_model_save_path=eval_path, log_path=eval_path, n_eval_episodes=EVAL_ENVS, eval_freq=EVAL_FREQ, deterministic=True, render=False, verbose=True, seed=EVAL_SEED)


    model = make_model(args.algo)
    #model.set_logger(configure(eval_path, ["stdout", "csv"]))
    model.set_logger(configure(eval_path, ["csv"]))

    model.learn(total_timesteps=TOTAL_TIME_STEPS, callback=eval_callback)

