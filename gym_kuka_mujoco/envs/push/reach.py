# Ensure we get the path separator correct on windows
import os
from gym_kuka_mujoco.envs.push import push_env
from gym import utils
MODEL_XML_PATH = 'R800_reach_gravity.xml'


class ReachEnv(push_env.PushEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'kuka_joint_1': 0.0,
            'kuka_joint_2': 0.712,
            'kuka_joint_3': 0.0,
            'kuka_joint_4': -1.26,
            'kuka_joint_5': 0.0,
            'kuka_joint_6': 1.17,
            'kuka_joint_7': 0.0
        }
        push_env.PushEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, n_substeps=20,
            target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.04,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

    def set_goal(self, goal):
        self.goal = goal

