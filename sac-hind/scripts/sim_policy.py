import argparse

import joblib
import tensorflow as tf
import numpy as np

from rllab.sampler.utils import tensor_utils

from rllab.envs.normalized_env import normalize
from rllab.envs.mujoco.ant_env import AntEnv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to the snapshot file.')
    parser.add_argument('--max-path-length', '-l', type=int, default=1000)
    parser.add_argument('--speedup', '-s', type=float, default=1)
    parser.add_argument('--deterministic', '-d', dest='deterministic',
                        action='store_true')
    parser.add_argument('--no-deterministic', '-nd', dest='deterministic',
                        action='store_false')
    parser.add_argument('--policy_h', type=int)
    parser.set_defaults(deterministic=True)

    args = parser.parse_args()

    return args


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    goal =  env._wrapped_env.model.data.qvel.flat[:]  + np.random.uniform(size=env._wrapped_env.model.nv, low=-15.0, high=15.0) 
    goal = goal.flat[:]
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o,goal)
        next_o, r, d, env_info = env.step(a)
        observations.append(env.observation_space.flatten(o))
        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
    if animated and not always_return_paths:
        return

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )



def simulate_policy(args):

    reward_list = []
    
    with tf.Session() as sess:
        data = joblib.load(args.file)
        if 'algo' in data.keys():
            policy = data['algo'].policy
            env = data['algo'].env
        else:
            policy = data['policy']
            env = data['env']
        with policy.deterministic(args.deterministic):
            #while True:
            for i in range(200):
                path = rollout(env, policy,
                               max_path_length=args.max_path_length,
                               animated=False, speedup=args.speedup,
                               always_return_paths=True)
                reward_list.append(path['rewards'].sum())
            total_returns = np.mean(reward_list)
            print (total_returns)
if __name__ == "__main__":
    args = parse_args()
    simulate_policy(args)
