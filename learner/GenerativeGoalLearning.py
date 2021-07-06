import time
import warnings
from itertools import count
from typing import Sequence, Tuple, Iterator

import gym
import numpy as np
import torch
from more_itertools import consume
from torch import Tensor

from learner.LSGAN import LSGAN
from learner.utils import print_message, display_goals

''' PARAMETERS '''
Rmin = 0.1
Rmax = 0.9

G_Input_Size = 4  # noise dim, somehow noise size is defined as 4 in their implementation for ant_gan experiment
G_Hidden_Size = 32
D_Hidden_Size = 64
GEN_VAR_COEFF = 0.001


Returns = Sequence[float]


def update_and_eval_policy(goals, agent, sampler):
    sampler.set_possible_goals(goals.numpy())
    sampler.reset()

    start = time.time()
    while True:
        episode_successes_per_goal = sampler.get_successes_of_goals()
        if all(len(sucs) >= 3 for g, sucs in episode_successes_per_goal.items()):
            break
        yield

    returns = [np.mean(episode_successes_per_goal[tuple(g)]) for g in goals.numpy()]
    return agent, returns


def label_goals(returns) -> Sequence[int]:
    """

    :rtype: Sequence[int]
    """
    return [int(Rmin <= r <= Rmax) for r in returns]


def sample(t: Tensor, k: int) -> Tensor:
    """
    https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    Implemented according to Appendix A.1: 2/3 from gan generated goals, 1/3 from old goals
    TODO: To avoid concentration of goals, concatinate only the goals which are away from old_goals
    """
    num_samples = min(len(t), k)
    indices = torch.randperm(len(t))[:num_samples]
    return t[indices]


def initialize_GAN(env: gym.GoalEnv) -> LSGAN:
    goalGAN = LSGAN(generator_input_size=G_Input_Size,
                    generator_hidden_size=G_Hidden_Size,
                    generator_output_size=dim_goal(env),
                    discriminator_input_size=dim_goal(env),
                    discriminator_hidden_size=D_Hidden_Size,
                    gen_variance_coeff=GEN_VAR_COEFF,
                    discriminator_output_size=1)  # distinguish whether g is in GOID or not
    return goalGAN


def dim_goal(env):
    return env.get_obs()["desired_goal"].shape[0]


def train_GAN(goals: Tensor, labels: Sequence[int], goalGAN):
    y: Tensor = torch.Tensor(labels).reshape(len(labels), 1)
    D = goalGAN.Discriminator.forward
    G = goalGAN.Generator.forward

    def D_loss_vec(z: Tensor) -> Tensor:
        return y * (D(goals) - 1) ** 2 + (1 - y) * (D(goals) + 1) ** 2 + (D(G(z)) + 1) ** 2

    iterations = 10
    for _ in range(iterations):
        '''
        Train Discriminator
        '''
        gradient_steps = 1
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Discriminator.zero_grad()
            D_loss = torch.mean(D_loss_vec(zs))
            D_loss.backward()
            goalGAN.D_Optimizer.step()
        '''
        Train Generator
        '''
        gradient_steps = 1
        β = goalGAN.Generator.variance_coeff
        for _ in range(gradient_steps):
            zs = torch.randn(len(labels), goalGAN.Generator.noise_size)
            goalGAN.Generator.zero_grad()
            G_loss = torch.mean(D(G(zs)) ** 2) + β / torch.var(G(zs), dim=0).mean()
            G_loss.backward()
            goalGAN.G_Optimizer.step()

    return goalGAN


def update_replay(goals: Tensor, goals_old: Tensor):
    if goals_old.shape[0] == 0:
        goals_old = goals[0][None]

    eps = 0.1
    for g in goals:
        g_is_close_to_goals_old = min((torch.dist(g, g_old) for g_old in goals_old)) < eps
        if not g_is_close_to_goals_old:
            goals_old = torch.cat((g[None], goals_old))
    return goals_old


def train_goalGAN(agent, goalGAN: LSGAN, sampler, pretrain_iters=5, use_old_goals=True) -> Iterator[
    None]:
    """
    Algorithm in the GAN paper, Florensa 2018

    for i in iterations:
        z         = sample_noise()                     # input for goal generator network
        goals     = G(z) union goals_old               # concat old goals with the generated ones
        π         = update_policy(goals, π)            # perform policy update, paper uses TRPO, Leon suggested to use PPO as it is simpler
        returns   = evaluate_policy(goals, π)          # needed to label the goals
        labels    = label_goals(goals)                 # needed to train discriminator network
        G, D      = train_GAN(goals, labels, G, D)
        goals_old = goals

    """

    #### PARAMETERS ####
    num_gan_goals = 60
    num_old_goals = num_gan_goals // 2 if use_old_goals else 0
    num_rand_goals = num_gan_goals // 2
    ####################


    start_pos = sampler.start_pos
    env = sampler.env
    # Initial training of the policy with random goals
    for iter_num in range(pretrain_iters):
        rand_goals = torch.clamp(
            torch.Tensor([start_pos]) + 0.05 * torch.randn(num_old_goals, dim_goal(env)), min=-100, max=100)
        agent, returns = yield from update_and_eval_policy(rand_goals, agent, sampler)
        labels = label_goals(returns)
        # display_goals(rand_goals.detach().numpy(), returns, iter_num, env, fileNamePrefix='_')
        goalGAN = train_GAN(rand_goals, labels, goalGAN)

    close_to_starting_pos = torch.Tensor([sampler.start_pos]) + 0.05 * torch.randn(num_old_goals, dim_goal(env))
    goals_old = torch.clamp(close_to_starting_pos, min=-100, max=100)

    for iter_num in count(pretrain_iters):
        z = torch.randn(size=(num_gan_goals, goalGAN.Generator.noise_size))
        raw_gan_goals = goalGAN.Generator.forward(z).detach()
        gan_goals = torch.clamp(raw_gan_goals + 0.1 * torch.randn(num_gan_goals, dim_goal(env)), min=-100, max=100)
        all_goals = torch.cat([gan_goals, sample(goals_old, k=num_old_goals)])
        agent, returns = yield from update_and_eval_policy(all_goals, agent, sampler)
        # display_goals(all_goals.detach().numpy(), returns, iter_num, env, gan_goals=raw_gan_goals.numpy())
        labels = label_goals(returns)
        if all([lab == 0 for lab in labels]): warnings.warn("All labels are 0")
        goalGAN = train_GAN(all_goals, labels, goalGAN)
        goals_old = update_replay(gan_goals, goals_old=goals_old)
