import copy
import numpy as np
from envs import make_env
from envs.utils import goal_distance
from algorithm.replay_buffer import Trajectory, goal_concat
from utils.gcc_utils import gcc_load_lib, c_double, c_int
from envs.distance_graph import DistanceGraph


# TODO: replaced goal_distance with get_graph_goal_distance

class TrajectoryPool:
    def __init__(self, args, pool_length):
        self.args = args
        self.length = pool_length

        self.pool = []
        self.pool_init_state = []
        self.counter = 0

    def insert(self, trajectory, init_state):
        if self.counter < self.length:
            self.pool.append(trajectory.copy())
            self.pool_init_state.append(init_state.copy())
        else:
            self.pool[self.counter % self.length] = trajectory.copy()
            self.pool_init_state[self.counter % self.length] = init_state.copy()
        self.counter += 1

    def pad(self):
        if self.counter >= self.length:
            return copy.deepcopy(self.pool), copy.deepcopy(self.pool_init_state)
        pool = copy.deepcopy(self.pool)
        pool_init_state = copy.deepcopy(self.pool_init_state)
        while len(pool) < self.length:
            pool += copy.deepcopy(self.pool)
            pool_init_state += copy.deepcopy(self.pool_init_state)
        return copy.deepcopy(pool[:self.length]), copy.deepcopy(pool_init_state[:self.length])


class MatchSampler:
    def __init__(self, args, achieved_trajectory_pool):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)
        self.dim = np.prod(self.env.reset()['achieved_goal'].shape)
        self.delta = self.env.distance_threshold

        self.length = args.episodes
        init_goal = self.env.reset()['achieved_goal'].copy()
        self.pool = np.tile(init_goal[np.newaxis, :], [self.length, 1]) + np.random.normal(0, self.delta,
                                                                                           size=(self.length, self.dim))
        self.init_state = self.env.reset()['observation'].copy()

        self.match_lib = gcc_load_lib('learner/cost_flow.c')
        self.achieved_trajectory_pool = achieved_trajectory_pool

        if self.args.graph:
            self.graph = args.graph

        # estimating diameter
        self.max_dis = 0
        for i in range(1000):
            obs = self.env.reset()
            dis = self.get_graph_goal_distance(obs['achieved_goal'], obs['desired_goal'])
            if dis > self.max_dis: self.max_dis = dis

    # Pre-computation of graph-based distances
    # def create_graph_distance(self):
    #     obstacles = list()
    #     field = self.env.env.env.adapt_dict["field"]
    #     obstacles = self.env.env.env.adapt_dict["obstacles"]
    #     num_vertices = self.args.num_vertices
    #     graph = DistanceGraph(args=self.args, field=field, num_vertices=num_vertices, obstacles=obstacles)
    #     graph.compute_cs_graph()
    #     graph.compute_dist_matrix()
    #     self.graph = graph

    def get_graph_goal_distance(self, goal_a, goal_b):
        if self.args.graph:
            d, _ = self.graph.get_dist(goal_a, goal_b)
            if d == np.inf:
                d = 9999
            return d
        else:
            return np.linalg.norm(goal_a - goal_b, ord=2)

    def get_foot_point(self, p, ls, le):
        d = ls - le
        u = (p[0] - ls[0]) * (ls[0] - le[0]) + (p[1] - ls[1]) * (ls[1] - le[1]) + (p[2] - ls[2]) * (ls[2] - le[2])
        u = u / (np.sum(np.square(d)))
        return ls + u * d

    def euler_dis(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))

    def get_route_goal_distance(self, goal_a, goal_b):
        r4 = np.array([1.3, 0.55, 0.4])
        if goal_a[1] < 0.75:
            return self.euler_dis(goal_a, r4)
        r1 = np.array([1.3, 0.95, 0.4])
        r2 = np.array([1.3, 0.95, 0.65])
        r3 = np.array([1.3, 0.55, 0.65])
        l1 = 0.25
        l2 = 0.4
        l3 = 0.25
        fp1 = np.array([1.3, 0.95, goal_a[2]])
        fp2 = np.array([1.3, goal_a[1], 0.65])
        if fp1[2] > r2[2]:
            d1 = self.euler_dis(goal_a, r2) + l2 + l3
        elif r2[2] >= fp1[2] >= r1[2]:
            d1 = self.euler_dis(goal_a, fp1) + np.abs(fp1[2] - r2[2]) + l2 + l3
        else:
            d1 = self.euler_dis(goal_a, r1) + l1 + l2 + l3

        if fp2[1] > r2[1]:
            d2 = self.euler_dis(goal_a, r2) + l2 + l3
        elif 0.75 <= fp2[1] <= r2[1]:
            d2 = self.euler_dis(goal_a, fp2) + np.abs(r3[1] - fp2[1]) + l3
        else:
            d2 = self.euler_dis(goal_a, r4)
        return min(d1, d2)

    def add_noise(self, pre_goal, noise_std=None):
        goal = pre_goal.copy()
        dim = 2 if self.args.env[:5] == 'fetch' else self.dim
        if noise_std is None: noise_std = self.delta
        goal[:dim] += np.random.normal(0, noise_std, size=dim)
        return goal.copy()

    def sample(self, idx):
        if self.args.env[:5] == 'fetch':
            return self.add_noise(self.pool[idx])
        else:
            return self.pool[idx].copy()

    def find(self, goal):
        res = np.sqrt(np.sum(np.square(self.pool - goal), axis=1))
        idx = np.argmin(res)
        if test_pool:
            self.args.logger.add_record('Distance/sampler', res[idx])
        return self.pool[idx].copy()

    def update(self, initial_goals, desired_goals):
        if self.achieved_trajectory_pool.counter == 0:
            self.pool = copy.deepcopy(desired_goals)
            return

        achieved_pool, achieved_pool_init_state = self.achieved_trajectory_pool.pad()
        candidate_goals = []
        candidate_edges = []
        candidate_id = []

        agent = self.args.agent
        achieved_value = []
        for i in range(len(achieved_pool)):
            obs = [goal_concat(achieved_pool_init_state[i], achieved_pool[i][j]) for j in
                   range(achieved_pool[i].shape[0])]
            feed_dict = {
                agent.raw_obs_ph: obs
            }
            value = agent.sess.run(agent.q_pi, feed_dict)[:, 0]
            value = np.clip(value, -1.0 / (1.0 - self.args.gamma), 0)
            achieved_value.append(value.copy())

        n = 0
        graph_id = {'achieved': [], 'desired': []}
        for i in range(len(achieved_pool)):
            n += 1
            graph_id['achieved'].append(n)
        for i in range(len(desired_goals)):
            n += 1
            graph_id['desired'].append(n)
        n += 1
        self.match_lib.clear(n)

        for i in range(len(achieved_pool)):
            self.match_lib.add(0, graph_id['achieved'][i], 1, 0)
        for i in range(len(achieved_pool)):
            for j in range(len(desired_goals)):

                # use graph_goal_distance here!
                if self.args.graph:
                    size = achieved_pool[i].shape[0]
                    res_1 = np.zeros(size)
                    for k in range(size):
                        res_1[k] = self.get_graph_goal_distance(achieved_pool[i][k], desired_goals[j])
                    res = res_1 - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
                elif self.args.route and self.args.env == 'FetchPickObstacle-v1':
                    size = achieved_pool[i].shape[0]
                    res_1 = np.zeros(size)
                    for k in range(size):
                        res_1[k] = self.get_route_goal_distance(achieved_pool[i][k], desired_goals[j])
                    res = res_1 - achieved_value[i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))
                else:
                    res = np.sqrt(np.sum(np.square(achieved_pool[i] - desired_goals[j]), axis=1)) - achieved_value[
                        i] / (self.args.hgg_L / self.max_dis / (1 - self.args.gamma))  # that was original

                match_dis = np.min(res) + goal_distance(achieved_pool[i][0], initial_goals[
                    j]) * self.args.hgg_c  # distance of initial positions: take l2 norm_as before
                match_idx = np.argmin(res)

                edge = self.match_lib.add(graph_id['achieved'][i], graph_id['desired'][j], 1, c_double(match_dis))
                candidate_goals.append(achieved_pool[i][match_idx])
                candidate_edges.append(edge)
                candidate_id.append(j)
        for i in range(len(desired_goals)):
            self.match_lib.add(graph_id['desired'][i], n, 1, 0)

        match_count = self.match_lib.cost_flow(0, n)
        assert match_count == self.length

        explore_goals = [0] * self.length
        for i in range(len(candidate_goals)):
            if self.match_lib.check_match(candidate_edges[i]) == 1:
                explore_goals[candidate_id[i]] = candidate_goals[i].copy()
        assert len(explore_goals) == self.length
        self.pool = np.array(explore_goals)


class HGGLearner:
    def __init__(self, args):
        self.args = args
        self.env = make_env(args)
        self.env_test = make_env(args)

        self.env_List = []
        for i in range(args.episodes):
            self.env_List.append(make_env(args))

        self.achieved_trajectory_pool = TrajectoryPool(args, args.hgg_pool_size)
        self.sampler = MatchSampler(args, self.achieved_trajectory_pool)

        self.stop_hgg_threshold = self.args.stop_hgg_threshold
        self.stop = False
        self.learn_calls = 0
        self.K = 1

    def learn(self, args, env, env_test, agent, buffer, write_goals=0):
        # Actual learning cycle takes place here!
        initial_goals = []
        desired_goals = []
        goal_list = []

        # get initial position and goal from environment for each epsiode
        for i in range(args.episodes):
            obs = self.env_List[i].reset()
            goal_a = obs['achieved_goal'].copy()
            goal_d = obs['desired_goal'].copy()
            initial_goals.append(goal_a.copy())
            desired_goals.append(goal_d.copy())

        # if HGG has not been stopped yet, perform crucial HGG update step here
        # by updating the sampler, a set of intermediate goals is provided and stored in sampler
        # based on distance to target goal distribution, similarity of initial states and expected reward (see paper)
        # by bipartite matching
        if not self.stop:
            self.sampler.update(initial_goals, desired_goals)
        if self.stop:
            buffer.stop_trade_off = True

        achieved_trajectories = []
        achieved_init_states = []

        explore_goals = []
        test_goals = []
        inside = []
        left_dis_total = 0
        for i in range(args.episodes):
            obs = self.env_List[i].get_obs()
            init_state = obs['observation'].copy()

            # if HGG has not been stopped yet, sample from the goals provided by the update step
            # if it has been stopped, the goal to explore is simply the one generated by the environment
            if not self.stop:
                explore_goal = self.sampler.sample(i)
            else:
                explore_goal = desired_goals[i]

            left_dis_total += self.sampler.get_graph_goal_distance(explore_goal, desired_goals[i])

            # store goals in explore_goals list to check whether goals are within goal space later
            explore_goals.append(explore_goal)
            test_goal = self.env.generate_goal()
            if test_goal.shape[-1] == 7:
                test_goal = test_goal[3:] # for some hand tasks
            test_goals.append(test_goal)

            # Perform HER training by interacting with the environment
            self.env_List[i].goal = explore_goal.copy()
            if write_goals != 0 and len(goal_list) < write_goals:
                goal_list.append(explore_goal.copy())
            obs = self.env_List[i].get_obs()
            current = Trajectory(obs)
            trajectory = [obs['achieved_goal'].copy()]
            for timestep in range(args.timesteps):
                # get action from the ddpg policy
                action = agent.step(obs, explore=True)
                # feed action to environment, get observation and reward
                obs, reward, done, info = self.env_List[i].step(action)
                trajectory.append(obs['achieved_goal'].copy())
                if timestep == args.timesteps - 1: done = True
                current.store_step(action, obs, reward, done)
                if done: break
            achieved_trajectories.append(np.array(trajectory))
            achieved_init_states.append(init_state)
            # Trajectory is stored in replay buffer, replay buffer can be normal or EBP
            buffer.store_trajectory(current)
            agent.normalizer_update(buffer.sample_batch())

            if buffer.steps_counter >= args.warmup:
                for _ in range(args.train_batches):
                    # train with Hindsight Goals (HER step)
                    info = agent.train(buffer.sample_batch())
                    args.logger.add_dict(info)
                # update target network
                agent.target_update()
        if left_dis_total == 0:
            buffer.dis_balance = 1000
            # maximum
        else:
            buffer.dis_balance = args.balance_eta * pow(2.71, (-left_dis_total / args.episodes) / (args.balance_sigma * args.balance_sigma))

        selection_trajectory_idx = {}
        for i in range(self.args.episodes):
            # only add trajectories with movement to the trajectory pool --> use default (L2) distance measure!
            if goal_distance(achieved_trajectories[i][0], achieved_trajectories[i][-1]) > 0.01:
                selection_trajectory_idx[i] = True
        for idx in selection_trajectory_idx.keys():
            self.achieved_trajectory_pool.insert(achieved_trajectories[idx].copy(), achieved_init_states[idx].copy())

        # unless in first call: Check which of the explore goals are inside the target goal space target goal space
        # is represented by a sample of test_goals directly generated from the environment an explore goal is
        # considered inside the target goal space, if it is closer than the distance_threshold to one of the test
        # goals (i.e. would yield a non-negative reward if that test goal was to be achieved)
        if self.learn_calls > 0:
            assert len(explore_goals) == len(test_goals)
            for ex in explore_goals:
                is_inside = 0
                for te in test_goals:
                    # TODO: check: originally with self.sampler.get_graph_goal_distance, now trying with goal_distance (L2)
                    if goal_distance(ex, te) <= self.env.env.env.distance_threshold:
                        is_inside = 1
                inside.append(is_inside)
            assert len(inside) == len(test_goals)
            inside_sum = 0
            for i in inside:
                inside_sum += i

            # If more than stop_hgg_threshold (e.g. 0.9) of the explore goals are inside the target goal space, stop HGG
            # and continue with normal HER.
            # By default, stop_hgg_threshold is disabled (set to a value > 1)
            average_inside = inside_sum / len(inside)
            self.args.logger.info("Average inside: {}".format(average_inside))
            if average_inside > self.stop_hgg_threshold:
                self.stop = True
                self.args.logger.info("Continue with normal HER")

        self.learn_calls += 1

        return goal_list if len(goal_list) > 0 else None
