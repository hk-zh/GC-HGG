import numpy as np
import time
from common import get_args,experiment_setup
from copy import deepcopy
import pickle
import os
import tensorflow as tf


def sample_hindsight_goals(buffer, num, path='hindsightGoals/temp'):
	if not os.path.exists(path):
		os.makedirs(path)
	output = open("{}/{}.pkl".format(path, num), "wb")
	l = []
	goals = buffer.sample_batch()['obs']
	for i in range(10):
		j = np.random.randint(len(goals))
		l.append(np.array([goals[j][3], goals[j][4], goals[j][5]]))
	pickle.dump(l, output)
	output.close()


if __name__=='__main__':

	# Getting arguments from command line + defaults
	# Set up learning environment including, gym env, ddpg agent, hgg/normal learner, tester
	args = get_args()
	env, env_test, agent, buffer, learner, tester, graph = experiment_setup(args)
	args.logger.summary_init(agent.graph, agent.sess)

	# Progress info
	args.logger.add_item('Epoch')
	args.logger.add_item('Cycle')
	args.logger.add_item('Episodes@green')
	args.logger.add_item('Timesteps')
	args.logger.add_item('TimeCost(sec)')

	best_success = -1

	# Algorithm info
	for key in agent.train_info.keys():
		args.logger.add_item(key, 'scalar')

	# Test info
	for key in tester.info:
		args.logger.add_item(key, 'scalar')

	args.logger.summary_setup()
	counter = 0
	rhg_cnt = 0
	# Learning
	for epoch in range(args.epoches):
		for cycle in range(args.cycles):
			args.logger.tabular_clear()
			args.logger.summary_clear()
			start_time = time.time()
			if args.rhg and epoch * args.cycles + cycle in [50, 100, 200, 300]:
				# sampled hindsight goals iteration 50, 100, 200, 300 will be recorded.
				rhg_cnt += 1
				sample_hindsight_goals(buffer, rhg_cnt, path="hindsightGoals/{}".format(args.env))

			# Learn
			goal_list = learner.learn(args, env, env_test, agent, buffer, write_goals=args.show_goals)

			# Log learning progresss
			tester.cycle_summary()
			args.logger.add_record('Epoch', str(epoch)+'/'+str(args.epoches))
			args.logger.add_record('Cycle', str(cycle)+'/'+str(args.cycles))
			args.logger.add_record('Episodes', buffer.counter)
			args.logger.add_record('Timesteps', buffer.steps_counter)
			args.logger.add_record('TimeCost(sec)', time.time()-start_time)

			# Save learning progress to progress.csv file
			args.logger.save_csv()

			args.logger.tabular_show(args.tag)
			args.logger.summary_show(buffer.counter)

			# Save latest policy
			policy_file = args.logger.my_log_dir + "saved_policy-latest"
			agent.saver.save(agent.sess, policy_file)

			# Save policy if new best_success was reached
			if args.logger.values["Success"] > best_success:
				best_success = args.logger.values["Success"]
				policy_file = args.logger.my_log_dir + "saved_policy-best"
				agent.saver.save(agent.sess, policy_file)
				args.logger.info("Saved as best policy to {}!".format(args.logger.my_log_dir))

		# Save periodic policy every epoch
		policy_file = args.logger.my_log_dir + "saved_policy"
		agent.saver.save(agent.sess, policy_file, global_step=epoch)
		args.logger.info("Saved periodic policy to {}!".format(args.logger.my_log_dir))

		# # Plot current goal distribution for visualization (G-HGG only)
		# if args.learn == 'hgg' and goal_list and args.show_goals != 0:
		# 	name = "{}goals_{}".format(args.logger.my_log_dir, epoch)
		# 	if args.graph:
		# 		learner.sampler.graph.plot_graph(goals=goal_list, save_path=name)
		# 	with open('{}.pkl'.format(name), 'wb') as file:
		# 			pickle.dump(goal_list, file)

		tester.epoch_summary()

	tester.final_summary()

