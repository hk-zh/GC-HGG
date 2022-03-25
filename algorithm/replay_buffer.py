import numpy as np
import copy
from envs.utils import quaternion_to_euler_angle
# from envs import make_env
from sklearn.neighbors import NearestNeighbors
import random


def goal_concat(obs, goal):
    return np.concatenate([obs, goal], axis=0)


def goal_based_process(obs):
    return goal_concat(obs['observation'], obs['desired_goal'])


class Trajectory:
    def __init__(self, init_obs):
        self.ep = {
            'obs': [copy.deepcopy(init_obs)],
            'rews': [],
            'acts': [],
            'done': []
        }
        self.length = 0

    def store_step(self, action, obs, reward, done):
        self.ep['acts'].append(copy.deepcopy(action))
        self.ep['obs'].append(copy.deepcopy(obs))
        self.ep['rews'].append(copy.deepcopy([reward]))
        self.ep['done'].append(copy.deepcopy([np.float32(done)]))
        self.length += 1

    def energy(self, env_id, w_potential=1.0, w_linear=1.0, w_rotational=1.0):
        # from "Energy-Based Hindsight Experience Prioritization"
        if env_id[:5] == 'Fetch' or env_id[:4] == 'Kuka':
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['achieved_goal'])
            obj = np.array([obj])

            clip_energy = 0.5
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            g, m, delta_t = 9.81, 1, 0.04
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_total = w_potential * potential_energy + w_linear * kinetic_energy
            energy_diff = np.diff(energy_total, axis=1)
            energy_transition = energy_total.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)
        else:
            assert env_id[:4] == 'Hand'
            obj = []
            for i in range(len(self.ep['obs'])):
                obj.append(self.ep['obs'][i]['observation'][-7:])
            obj = np.array([obj])

            clip_energy = 2.5
            g, m, delta_t, inertia = 9.81, 1, 0.04, 1
            quaternion = obj[:, :, 3:].copy()
            angle = np.apply_along_axis(quaternion_to_euler_angle, 2, quaternion)
            diff_angle = np.diff(angle, axis=1)
            angular_velocity = diff_angle / delta_t
            rotational_energy = 0.5 * inertia * np.power(angular_velocity, 2)
            rotational_energy = np.sum(rotational_energy, axis=2)
            obj = obj[:, :, :3]
            height = obj[:, :, 2]
            height_0 = np.repeat(height[:, 0].reshape(-1, 1), height[:, 1::].shape[1], axis=1)
            height = height[:, 1::] - height_0
            potential_energy = g * m * height
            diff = np.diff(obj, axis=1)
            velocity = diff / delta_t
            kinetic_energy = 0.5 * m * np.power(velocity, 2)
            kinetic_energy = np.sum(kinetic_energy, axis=2)
            energy_total = w_potential * potential_energy + w_linear * kinetic_energy + w_rotational * rotational_energy
            energy_diff = np.diff(energy_total, axis=1)
            energy_transition = energy_total.copy()
            energy_transition[:, 1::] = energy_diff.copy()
            energy_transition = np.clip(energy_transition, 0, clip_energy)
            energy_transition_total = np.sum(energy_transition, axis=1)
            energy_final = energy_transition_total.reshape(-1, 1)
            return np.sum(energy_final)


class ReplayBuffer_Episodic:
    def __init__(self, args):
        self.args = args
        if args.buffer_type == 'energy':
            self.energy = True
            self.energy_sum = 0.0
            self.energy_offset = 0.0
            self.energy_max = 1.0
        else:
            self.energy = False
        if args.graph:
            self.graph = args.graph
        # env = make_env(args)
        # self.workspace = env.get_workspace()
        self.buffer = {}
        self.achieved_goals = []
        self.steps = []
        self.length = 0
        self.counter = 0
        self.steps_counter = 0
        self.dis_balance = 0
        self.iter_balance = 1
        self.eta = self.args.balance_eta
        self.sigma = self.args.balance_sigma
        self.tau = self.args.balance_tau
        self.stop_trade_off = False
        self.ignore = True

        if args.curriculum:
            if args.learn == "normal":
                self.sample_batch = self.lazier_and_goals_sample_kg
            elif args.learn == "hgg":
                self.sample_batch = self.sample_batch_diversity_proximity_trade_off
        else:
            self.sample_batch = self.sample_batch_ddpg

    def get_goal_distance(self, goal_a, goal_b):
        d, _ = self.graph.get_dist(goal_a, goal_b)
        if d == np.inf:
            d = 9999
        return d

    def get_goal_distance_grid(self, goal_a, goal_b):
        d = self.graph.get_dist_grid(goal_a, goal_b)
        if d == np.inf:
            d = 9999
        return d

    def update_dis_balance(self, avg_dis):
        self.dis_balance = self.eta * np.exp((-avg_dis) / (self.sigma * self.sigma))

    def update_iter_balance(self):
        self.iter_balance *= (1 + self.tau)

    def store_trajectory(self, trajectory):
        episode = trajectory.ep
        energy = None
        if self.energy:
            energy = trajectory.energy(self.args.env)
            self.energy_sum += energy
        if self.counter == 0:
            for key in episode.keys():
                self.buffer[key] = []
            if self.energy:
                self.buffer_energy = []
                self.buffer_energy_sum = []
        if self.counter < self.args.buffer_size:
            for key in self.buffer.keys():
                self.buffer[key].append(episode[key])
            if self.energy:
                self.buffer_energy.append(copy.deepcopy(energy))
                self.buffer_energy_sum.append(copy.deepcopy(self.energy_sum))
            self.length += 1
            self.steps.append(trajectory.length)
        else:
            idx = self.counter % self.args.buffer_size
            for key in self.buffer.keys():
                self.buffer[key][idx] = episode[key]
            if self.energy:
                self.energy_offset = copy.deepcopy(self.buffer_energy_sum[idx])
                self.buffer_energy[idx] = copy.deepcopy(energy)
                self.buffer_energy_sum[idx] = copy.deepcopy(self.energy_sum)
            self.steps[idx] = trajectory.length
        self.counter += 1
        self.steps_counter += trajectory.length

    def energy_sample(self):
        t = self.energy_offset + np.random.uniform(0, 1) * (self.energy_sum - self.energy_offset)
        if self.counter > self.args.buffer_size:
            if self.buffer_energy_sum[-1] >= t:
                return self.energy_search(t, self.counter % self.length, self.length - 1)
            else:
                return self.energy_search(t, 0, self.counter % self.length - 1)
        else:
            return self.energy_search(t, 0, self.length - 1)

    def energy_search(self, t, l, r):
        if l == r:
            return l
        mid = (l + r) // 2
        if self.buffer_energy_sum[mid] >= t:
            return self.energy_search(t, l, mid)
        else:
            return self.energy_search(t, mid + 1, r)

    def sample_batch_ddpg(self, batch_size=-1, normalizer=False, plain=False):
        assert int(normalizer) + int(plain) <= 1
        if batch_size == -1:
            batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])

        for i in range(batch_size):
            if self.energy:
                idx = self.energy_sample()
            else:
                idx = np.random.randint(self.length)
            step = np.random.randint(self.steps[idx])

            if self.args.goal_based:
                if plain:
                    # no additional tricks
                    goal = self.buffer['obs'][idx][step]['desired_goal']
                elif normalizer:
                    # uniform sampling for normalizer update
                    goal = self.buffer['obs'][idx][step]['achieved_goal']
                else:
                    # upsampling by HER trick
                    if (self.args.her != 'none') and (np.random.uniform() <= self.args.her_ratio):
                        if self.args.her == 'match':
                            goal = self.args.goal_sampler.sample()
                            goal_pool = np.array([obs['achieved_goal'] for obs in self.buffer['obs'][idx][step + 1:]])
                            step_her = (step + 1) + np.argmin(np.sum(np.square(goal_pool - goal), axis=1))
                            goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                        else:
                            step_her = {
                                'final': self.steps[idx],
                                'future': np.random.randint(step + 1, self.steps[idx] + 1)
                            }[self.args.her]
                            goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                    else:
                        goal = self.buffer['obs'][idx][step]['desired_goal']

                achieved = self.buffer['obs'][idx][step + 1]['achieved_goal']
                achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
                obs = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
                obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], goal)
                act = self.buffer['acts'][idx][step]
                rew = self.args.compute_reward((achieved, achieved_old), goal)
                done = self.buffer['done'][idx][step]

                batch['obs'].append(copy.deepcopy(obs))
                batch['obs_next'].append(copy.deepcopy(obs_next))
                batch['acts'].append(copy.deepcopy(act))
                batch['rews'].append(copy.deepcopy([rew]))
                batch['done'].append(copy.deepcopy(done))
            else:
                for key in ['obs', 'acts', 'rews', 'done']:
                    if key == 'obs':
                        batch['obs'].append(copy.deepcopy(self.buffer[key][idx][step]))
                        batch['obs_next'].append(copy.deepcopy(self.buffer[key][idx][step + 1]))
                    else:
                        batch[key].append(copy.deepcopy(self.buffer[key][idx][step]))

        return batch

    @staticmethod
    def fa(k, a_set, v_set, sim, row, col):
        if len(a_set) == 0:
            init_a_set = []
            marginal_v = 0
            for i in v_set:
                max_ki = 0
                if k == col[i]:
                    max_ki = sim[i]
                init_a_set.append(max_ki)
                marginal_v += max_ki
            return marginal_v, init_a_set

        new_a_set = []
        marginal_v = 0
        for i in v_set:
            sim_ik = 0
            if k == col[i]:
                sim_ik = sim[i]

            if sim_ik > a_set[i]:
                max_ki = sim_ik
                new_a_set.append(max_ki)
                marginal_v += max_ki - a_set[i]
            else:
                new_a_set.append(a_set[i])
        return marginal_v, new_a_set

    def lazier_and_goals_sample_kg(self):
        if self.length <= 2 * self.args.batch_size or self.stop_trade_off:
            return self.sample_batch_ddpg()
        batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])
        goals = []
        ac_goals = []
        experience_buffer = []
        # still use her to select goals in each episode
        for idx in range(self.length):
            step = np.random.randint(self.steps[idx])
            step_her = np.random.randint(step + 1, self.steps[idx] + 1)
            if np.random.uniform() <= self.args.her_ratio:
                goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                experience_buffer.append([idx, step, goal])
            else:
                goal = self.buffer['obs'][idx][step]['desired_goal']
                experience_buffer.append([idx, step, goal])
            goals.append(goal)
            ac_goals.append(self.buffer['obs'][idx][step]['achieved_goal'])

        num_neighbor = 1
        kgraph = NearestNeighbors(
            n_neighbors=num_neighbor, algorithm='kd_tree',
            metric='euclidean').fit(goals).kneighbors_graph(
            mode='distance').tocoo(copy=False)
        row = kgraph.row
        col = kgraph.col
        sim = np.exp(
            -np.divide(np.power(kgraph.data, 2),
                       np.mean(kgraph.data) ** 2))

        sel_idx_set = []
        idx_set = [i for i in range(len(goals))]
        balance = self.iter_balance
        v_set = [i for i in range(len(goals))]
        max_set = []
        for i in range(batch_size):
            sub_size = 3
            sub_set = random.sample(idx_set, sub_size)
            sel_idx = -1
            max_marginal = float("-inf")
            for j in range(sub_size):
                k_idx = sub_set[j]
                marginal_v, new_a_set = self.fa(k_idx, max_set, v_set, sim, row,
                                                col)
                euc = np.linalg.norm(goals[sub_set[j]] - ac_goals[sub_set[j]])
                marginal_v = marginal_v - balance * euc
                if marginal_v > max_marginal:
                    sel_idx = k_idx
                    max_marginal = marginal_v
                    max_set = new_a_set

            idx_set.remove(sel_idx)
            sel_idx_set.append(sel_idx)
        for i in sel_idx_set:
            idx = experience_buffer[i][0]
            step = experience_buffer[i][1]
            goal = experience_buffer[i][2]
            achieved = self.buffer['obs'][idx][step + 1]['achieved_goal']
            achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
            obs = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
            obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], goal)
            act = self.buffer['acts'][idx][step]
            rew = self.args.compute_reward((achieved, achieved_old), goal)
            done = self.buffer['done'][idx][step]

            batch['obs'].append(copy.deepcopy(obs))
            batch['obs_next'].append(copy.deepcopy(obs_next))
            batch['acts'].append(copy.deepcopy(act))
            batch['rews'].append(copy.deepcopy([rew]))
            batch['done'].append(copy.deepcopy(done))
        return batch

    def compute_diversity_graph(self, batch):
        goals = []
        diversity = 0.0
        for i in range(len(batch)):
            idx = batch[i][0]
            step = batch[i][1]
            ac_goal = self.buffer['obs'][idx][step]['achieved_goal']
            goals.append(ac_goal)

        num_neighbor = 1
        kgraph = NearestNeighbors(
            n_neighbors=num_neighbor, algorithm='kd_tree',
            metric='euclidean').fit(goals).kneighbors_graph(
            mode='distance').tocoo(copy=False)
        row = kgraph.row
        col = kgraph.col
        n = len(row)
        cnt = 0
        for i in range(n):
            dis = self.get_goal_distance(goals[row[i]], goals[col[i]])
            if dis != 9999:
                diversity += (dis + kgraph.data[i]) / 2
                cnt += 1
        return diversity / cnt

    def compute_diversity2(self, batch):
        goals = []
        diversity = 0.0
        for i in range(len(batch)):
            idx = batch[i][0]
            step = batch[i][1]
            ac_goal = self.buffer['obs'][idx][step]['achieved_goal']
            goals.append(ac_goal)

        num_neighbor = 1
        kgraph = NearestNeighbors(
            n_neighbors=num_neighbor, algorithm='kd_tree',
            metric='euclidean').fit(goals).kneighbors_graph(
            mode='distance').tocoo(copy=False)

        for i in range(len(kgraph.data)):
            diversity += kgraph.data[i]
        return diversity / len(batch)

    def compute_proximity_graph(self, batch):
        proximity = 0.0
        cnt = 0
        for i in range(len(batch)):
            idx = batch[i][0]
            step = batch[i][1]
            goal = batch[i][2]
            ac_goal = self.buffer['obs'][idx][step]['achieved_goal']
            dis = self.get_goal_distance(ac_goal, goal)
            if dis != 9999:
                proximity = proximity + dis
                cnt += 1
        if cnt == 0:
            return 0
        else:
            return proximity / cnt

    def compute_proximity(self, batch):
        proximity = 0.0
        for i in range(len(batch)):
            idx = batch[i][0]
            step = batch[i][1]
            goal = batch[i][2]
            ac_goal = self.buffer['obs'][idx][step]['achieved_goal']
            proximity = proximity + np.linalg.norm(ac_goal - goal)
        return proximity / len(batch)

    def sample_batch_diversity_proximity_trade_off(self):
        if self.stop_trade_off:
            return self.sample_batch_ddpg()

        batch_size = self.args.batch_size
        batch = dict(obs=[], obs_next=[], acts=[], rews=[], done=[])
        batches = []
        N = self.args.K
        for i in range(N):
            batches.append([])
        sel_batch = None
        F_max = float('-inf')

        for i in range(N):
            for j in range(batch_size):
                idx = self.energy_sample()
                step = np.random.randint(self.steps[idx])
                step_her = np.random.randint(step + 1, self.steps[idx] + 1)
                if np.random.uniform() <= self.args.her_ratio:
                    goal = self.buffer['obs'][idx][step_her]['achieved_goal']
                    batches[i].append([idx, step, goal])
                else:
                    goal = self.buffer['obs'][idx][step]['desired_goal']
                    batches[i].append([idx, step, goal])

            if self.args.graph:
                diversity = self.compute_diversity_graph(batches[i])
                proximity = self.compute_proximity_graph(batches[i])
            else:
                diversity = self.compute_diversity2(batches[i])
                proximity = self.compute_proximity(batches[i])

            lamb = self.dis_balance
            F = diversity - lamb * proximity
            if F > F_max:
                F_max = F
                sel_batch = batches[i]

        for i in range(batch_size):
            idx = sel_batch[i][0]
            step = sel_batch[i][1]
            goal = sel_batch[i][2]
            achieved = self.buffer['obs'][idx][step + 1]['achieved_goal']
            achieved_old = self.buffer['obs'][idx][step]['achieved_goal']
            obs = goal_concat(self.buffer['obs'][idx][step]['observation'], goal)
            obs_next = goal_concat(self.buffer['obs'][idx][step + 1]['observation'], goal)
            act = self.buffer['acts'][idx][step]
            rew = self.args.compute_reward((achieved, achieved_old), goal)
            done = self.buffer['done'][idx][step]

            batch['obs'].append(copy.deepcopy(obs))
            batch['obs_next'].append(copy.deepcopy(obs_next))
            batch['acts'].append(copy.deepcopy(act))
            batch['rews'].append(copy.deepcopy([rew]))
            batch['done'].append(copy.deepcopy(done))
        return batch
