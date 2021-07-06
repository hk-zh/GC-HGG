import numpy as np
import gym
from envs.utils import goal_distance, goal_distance_obs
from utils.os_utils import remove_color
from gym_kuka_mujoco.envs import *
from gym.wrappers.time_limit import TimeLimit




def make_env(envName):
    env = {
        'KukaReach-v1': ReachEnv,
        'KukaPickAndPlaceObstacle-v1': PickObstacleEnv,
        'KukaPickAndPlaceObstacle-v2': PickObstacleEnvV2,
        'KukaPickNoObstacle-v1': PickNoObstacleEnv,
        'KukaPickNoObstacle-v2': PickNoObstacleEnvV2,
        'KukaPickThrow-v1': PickThrowEnv,
        'KukaPushLabyrinth-v1': PushLabyrinthEnv,
        'KukaPushLabyrinth-v2': PushLabyrinthEnvV2,
        'KukaPushSlide-v1': PushSlide,
        'KukaPushNew-v1': PushNewEnv
    }[envName]()
    MAXEPISODESTEPS = {
        'KukaReach-v1': 50,
        'KukaPickAndPlaceObstacle-v1': 100,
        'KukaPickAndPlaceObstacle-v2': 100,
        'KukaPickNoObstacle-v1': 100,
        'KukaPickNoObstacle-v2': 100,
        'KukaPickThrow-v1': 100,
        'KukaPushLabyrinth-v1': 100,
        'KukaPushSlide-v1': 100,
        'KukaPushNew-v1': 100,
        'KukaPushLabyrinth-v2': 100
    }[envName]
    env = TimeLimit(env, max_episode_steps=MAXEPISODESTEPS)
    return env


class KukaEnvWrapper():
    def __init__(self, args):
        self.args = args
        self.env = make_env(args.env)
        self.np_random = self.env.env.np_random
        self.distance_threshold = self.env.env.distance_threshold
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.max_episode_steps = self.env._max_episode_steps
        self.fixed_obj = False
        self.has_object = self.env.env.has_object
        self.obj_range = self.env.env.obj_range
        # self.target_range = self.env.env.target_range
        self.target_offset = self.env.env.target_offset
        self.target_in_the_air = self.env.env.target_in_the_air
        if self.has_object: self.height_offset = self.env.env.height_offset

        self.render = self.env.render
        self.get_obs = self.env.env._get_obs
        self.reset_sim = self.env.env._reset_sim

        self.reset_ep()
        self.env_info = {
            'Rewards': self.process_info_rewards,  # episode cumulative rewards
            'Distance': self.process_info_distance,  # distance in the last step
            'Success@green': self.process_info_success  # is_success in the last step
        }
        self.env.reset()
        self.fixed_obj = True

    def compute_reward(self, achieved, goal):
        return self.env.env.compute_reward(achieved[0], goal, None)

    def compute_distance(self, achieved, goal):
        return np.sqrt(np.sum(np.square(achieved - goal)))

    def process_info_rewards(self, obs, reward, info):
        self.rewards += reward
        return self.rewards

    def process_info(self, obs, reward, info):
        return {
            remove_color(key): value_func(obs, reward, info)
            for key, value_func in self.env_info.items()
        }

    def process_info_distance(self, obs, reward, info):
        return self.compute_distance(obs['achieved_goal'], obs['desired_goal'])

    def process_info_success(self, obs, reward, info):
        return info['is_success']

    def step(self, action):
        # imaginary infinity horizon (without done signal)
        obs, reward, done, info = self.env.step(action)
        info = self.process_info(obs, reward, info)
        reward = self.compute_reward((obs['achieved_goal'], self.last_obs['achieved_goal']),
                                     obs['desired_goal'])  # TODO: why the heck second argument if it is then ignored??
        self.last_obs = obs.copy()
        return obs, reward, False, info

    def reset_ep(self):
        self.rewards = 0.0

    def reset(self):
        self.reset_ep()
        self.last_obs = (self.env.reset()).copy()
        return self.last_obs.copy()

    @property
    def sim(self):
        return self.env.env.sim

    @sim.setter
    def sim(self, new_sim):
        self.env.env.sim = new_sim

    @property
    def initial_state(self):
        return self.env.env.initial_state

    @property
    def initial_gripper_xpos(self):
        return self.env.env.initial_gripper_xpos.copy()

    @property
    def goal(self):
        return self.env.env.goal.copy()

    @goal.setter
    def goal(self, value):
        self.env.env.goal = value.copy()

    def generate_goal(self):
        return self.env.env._sample_goal()

    def reset(self):
        self.reset_ep()
        self.env.env._reset_sim()
        self.goal = self.generate_goal()
        self.last_obs = (self.get_obs()).copy()
        return self.get_obs()

    def set_goal(self, goal):
        self.env.env.set_goal(goal)

    def stepJoints(self, joints):
        self.env.env.stepJoints(joints)

