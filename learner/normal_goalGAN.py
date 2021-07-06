import numpy as np
from envs import make_env
from algorithm.replay_buffer import Trajectory
from learner.GenerativeGoalLearning import train_goalGAN, initialize_GAN
from typing import Tuple, Mapping, List
from itertools import cycle
from algorithm import create_agent
GoalHashable = Tuple[float]


class MatchSampler:
    def __init__(self, args, env, use_random_starting_pos=False):
        self.args = args
        self.env = env
        self.delta = self.env.distance_threshold
        self.init_state = self.env.reset()['observation'].copy()
        self.possible_goals = None
        self.successes_per_goal: Mapping[GoalHashable, List[bool]] = dict()
        self.use_random_starting_pos = use_random_starting_pos
        self.start_pos = self.env.get_obs()['achieved_goal'].copy()
        self.agent_pos = None
        self.step_num = 0

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env[:5] == 'fetch' else self.dim
        if noise_std is None: noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self):
        return next(self.possible_goals).copy()

    def new_initial_pos(self):
        if not self.use_random_starting_pos:
            return self.start_pos

    def reset(self):
        obs = self.env.reset()
        self.start_pos = obs['achieved_goal'].copy()
        self.env.goal = self.sample().copy()
        self.step_num = 0

    def set_possible_goals(self, goals, entire_space=False) -> None:
        if goals is None and entire_space:
            self.possible_goals = None
            self.successes_per_goal = dict()
            return
        self.possible_goals = cycle(np.random.permutation(goals))
        self.successes_per_goal = {tuple(g): [] for g in goals}

    def get_successes_of_goals(self) -> Mapping[GoalHashable, List[bool]]:
        return dict(self.successes_per_goal)

    def step(self):
        self.step_num += 1


class NormalGoalGANLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.sampler = MatchSampler(args, self.env)
        self.loop = None

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        if self.loop is None:
            self.loop = train_goalGAN(agent, initialize_GAN(env=self.env), self.sampler, 5, True)
        for _ in range(args.episodes):
            obs = self.env.get_obs()
            current = Trajectory(obs)
            next(self.loop)
            explore_goal = self.sampler.sample()
            self.env.goal = explore_goal.copy()
            has_success = False
            for timestep in range(args.timesteps):
                action = agent.step(obs, explore=True)
                obs, reward, done, _ = self.env.step(action)
                is_success = reward == 0
                current.store_step(action, obs, reward, done)
                if is_success and not has_success:
                    has_success = True
                    if len(self.sampler.successes_per_goal) > 0:
                        self.sampler.successes_per_goal[tuple(self.env.goal)].append(is_success)
                if timestep == args.timesteps - 1 and not has_success:
                    if len(self.sampler.successes_per_goal) > 0:
                        self.sampler.successes_per_goal[tuple(self.env.goal)].append(is_success)
                if done: break
            next(self.loop)
            self.sampler.reset()
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                agent.target_update()