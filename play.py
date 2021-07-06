import numpy as np
from envs import make_env
from algorithm.replay_buffer import goal_based_process
from utils.os_utils import make_dir
from common import get_args
import re
from ast import literal_eval
import tensorflow as tf
import os
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from resource import sample_trajectory


class Player:
    def __init__(self, args):
        # initialize environment
        self.args = args
        self.env = make_env(args)
        self.args.timesteps = self.env.max_episode_steps
        self.env_test = make_env(args)
        self.info = []
        self.test_rollouts = 100

        # get current policy from path (restore tf session + graph)
        self.play_dir = args.play_path
        self.play_epoch = args.play_epoch
        self.meta_path = os.path.join(self.play_dir, "saved_policy-{}.meta".format(self.play_epoch))
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(self.meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(self.play_dir))
        graph = tf.get_default_graph()
        self.raw_obs_ph = graph.get_tensor_by_name("raw_obs_ph:0")
        self.pi = graph.get_tensor_by_name("main/policy/net/pi/Tanh:0")

    def my_step_batch(self, obs):
        # compute actions from obs based on current policy by running tf session initialized before
        actions = self.sess.run(self.pi, {self.raw_obs_ph: obs})
        return actions

    def play(self):
        # play policy on env
        env = self.env
        acc_sum, obs = 0.0, []
        for i in range(self.test_rollouts):
            obs.append(goal_based_process(env.reset()))
            for timestep in range(self.args.timesteps):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                env.render()


    def demoRecordReach(self, raw_path="videos/KukaReach1"):
        env = self.env
        test_rollouts = 5
        goals = [[0.84604588, 0.14732964, 1.35766576], [0.79483348, -0.14184732, 1.20930532],
                 [0.919015, -0.15907337, 1.18060975], [7.11554270e-01, 1.51756884e-03, 1.34433537e+00],
                 [0.70905836, 0.13042637, 1.19320888]]
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        for i in range(test_rollouts):
            env.reset()
            env.set_goal(np.array(goals[i]))
            print("Rollout {}/{} ...".format(i + 1, test_rollouts))
            for timestep in range(len(sample_trajectory[i])):
                if timestep == 0: continue
                action = np.array(sample_trajectory[i][timestep]) - np.array(sample_trajectory[i][timestep-1])
                zero = np.zeros(1)
                action = np.concatenate([action * 22, zero])
                env.step(action)
                recorder.capture_frame()
        recorder.close()

    def demoRecordPush(self, raw_path="videos/KukaPush"):
        env = self.env
        test_rollouts = 5
        goals = [[0.68, -0.18, 0.85], [0.60, -0.3, 0.85], [0.72, -0.28, 0.85], [0.58, -0.3, 0.85], [0.62, -0.25, 0.85]]
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        acc_sum, obs = 0.0, []
        test_rollouts = 5
        for i in range(test_rollouts):
            env.reset()
            env.set_goal(np.array(goals[i]))
            obs.append(goal_based_process(env.get_obs()))
            print("Rollout {}/{} ...".format(i + 1, test_rollouts))
            for timestep in range(self.args.timesteps):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                recorder.capture_frame()
        recorder.close()

    def demoRecordPickNoObstacle(self, raw_path="videos/KukaPickNoObstacle"):
        env = self.env
        test_rollouts = 5
        goals = [[0.80948876, -0.24847823, 1.15], [0.90204398, -0.24176245, 1.15], [0.72934716, -0.19637749, 1.15], [0.8429464, -0.20765762, 1.15], [0.6970663, -0.18643907, 1.15]]
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        acc_sum, obs = 0.0, []
        test_rollouts = 5
        for i in range(test_rollouts):
            env.reset()
            env.set_goal(np.array(goals[i]))
            obs.append(goal_based_process(env.get_obs()))
            print("Rollout {}/{} ...".format(i + 1, test_rollouts))
            for timestep in range(200):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                recorder.capture_frame()
        recorder.close()

    def demoRecordPickAndPlaceObstacle(self, raw_path="videos/KukaPickAndPlaceObstacle"):
        env = self.env
        test_rollouts = 5
        goals = [[0.80948876, -0.24847823, 0.85], [0.90204398, -0.24176245, 0.85], [0.72934716, -0.19637749, 0.85], [0.6970663, -0.25643907, 0.85], [0.7029464, -0.18765762, 0.85]]
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        acc_sum, obs = 0.0, []
        test_rollouts = 5
        for i in range(test_rollouts):
            env.reset()
            env.set_goal(np.array(goals[i]))
            obs.append(goal_based_process(env.get_obs()))
            print("Rollout {}/{} ...".format(i + 1, test_rollouts))
            for timestep in range(200):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                recorder.capture_frame()
        recorder.close()

    def record_video(self, raw_path="myrecord"):
        env = self.env
        test_rollouts = 5
        # play policy on env
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        acc_sum, obs = 0.0, []
        for i in range(test_rollouts):
            obs.append(goal_based_process(env.reset()))
            if hasattr(env.env.env, "set_camera_pos"):
                env.env.env.set_camera_pos(i)
            print("Rollout {}/{} ...".format(i + 1, test_rollouts))
            for timestep in range(self.args.timesteps):
                actions = self.my_step_batch(obs)
                obs, infos = [], []
                ob, _, _, info = env.step(actions[0])
                obs.append(goal_based_process(ob))
                infos.append(info)
                recorder.capture_frame()
            print("... done.")
        recorder.close()

    def demoRecordReachJoints(self, raw_path="videos/KukaReachJoints"):
        file = open("KukaPushJointsTrajectory.txt", 'r')
        env = self.env
        test_rollouts = 5
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        goals = [[0.84604588, 0.14732964, 1.35766576], [0.79483348, -0.14184732, 1.20930532],
                 [0.919015, -0.15907337, 1.18060975], [7.11554270e-01, 1.51756884e-03, 1.34433537e+00],
                 [0.70905836, 0.13042637, 1.19320888]]
        for i in range(test_rollouts):
            jointsTrajectory = re.sub(r"([^[])\s+([^]])", r"\1 \2", file.readline())
            jointsTrajectory = np.array(literal_eval(jointsTrajectory))
            env.reset()
            env.set_goal(np.array(goals[i]))
            for j in range(len(jointsTrajectory)):
                self.env.stepJoints(jointsTrajectory[j])
                recorder.capture_frame()
            print("... done.")
        recorder.close()

    def demoRecordPushJoints(self, raw_path="videos/KukaPushJoints"):
        file = open("KukaPushJointsTrajectory2.txt", 'r')
        env = self.env
        test_rollouts = 5
        goals = [[0.68, -0.18, 0.85], [0.60, -0.3, 0.85], [0.72, -0.28, 0.85], [0.58, -0.3, 0.85], [0.62, -0.25, 0.85]]
        recorder = VideoRecorder(env.env.env, base_path=raw_path)
        jointsTrajectory = re.sub(r"([^[])\s+([^]])", r"\1 \2", file.readline())
        jointsTrajectory = np.array(literal_eval(jointsTrajectory))
        env.reset()
        env.set_goal(np.array(goals[1]))
        for j in range(len(jointsTrajectory)):
            self.env.stepJoints(jointsTrajectory[j])
            recorder.capture_frame()

        print("... done.")





if __name__ == "__main__":
    # Call play.py in order to see current policy progress
    args = get_args()
    player = Player(args)
    if not args.record:
        player.play()
    else:
        player.record_video(raw_path="figures/" + args.play_path[8:])
    # player.demoRecordPush()
    # player.demoRecordPushJoints()
    # player.demoRecordReach()
    # player.demoRecordReach1()
    # player.demoRecordPickNoObstacle()
    # player.demoRecordPickAndPlaceObstacle()

