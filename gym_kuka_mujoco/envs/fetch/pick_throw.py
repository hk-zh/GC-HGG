import numpy as np
from gym import utils as gym_utils
from gym_kuka_mujoco.envs.fetch import fetch_env
from gym_kuka_mujoco.envs import rotations, utils

MODEL_XML_PATH = 'R800_pick_and_throw_gravity.xml'


class PickThrowEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'kuka_joint_1': 0.0,
            'kuka_joint_2': 0.921,
            'kuka_joint_3': 0.0,
            'kuka_joint_4': -1.42,
            'kuka_joint_5': 0.0,
            'kuka_joint_6': 0.796,
            'kuka_joint_7': 0.0,
            'r_gripper_finger_joint': 0.0,
            'l_gripper_finger_joint': 0.0
        }
        self.target_range_x = 0.06  # entire table: 0.125
        self.target_range_y = 0.06  # entire table: 0.175

        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.0-0.05, 0, 1.2, 0.425, 0.35, 0.2]
        self.adapt_dict["obstacles"] = [[1.085-0.05, 0, 1.05, 0.01, 0.35, 0.05], [1.415-0.05, 0, 1.05, 0.01, 0.35, 0.05], [1.25-0.05, 0.34, 1.05, 0.175, 0.01, 0.05], [1.25-0.05, -0.34, 1.05, 0.175, 0.01, 0.05], [1.25-0.05, 0, 1.05, 0.01, 0.35, 0.05], [1.25-0.05, 0.17, 1.05, 0.175, 0.01, 0.05], [1.25-0.05, 0, 1.05, 0.175, 0.01, 0.05], [1.25-0.05, -0.17, 1.05, 0.175, 0.01, 0.05]]

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, n_substeps=20,
            target_in_the_air=True, target_offset=0.0,
            obj_range=0.06, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        gym_utils.EzPickle.__init__(self)


    def _sample_goal(self):

        index = int(np.floor(self.np_random.uniform(0, 8)))
        goal = self.targets[index]

        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.init_center[:2]
            object_xpos = self.init_center[:2] + self.np_random.uniform(-self.obj_range, self.obj_range,
                                                                                 size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        # Move end effector into position.
        target = self.sim.data.get_site_xpos('gripper_tip').copy()
        rotation = self.sim.data.get_body_xquat('gripper_tip').copy()
        self.sim.data.set_mocap_pos('kuka_mocap', target)
        self.sim.data.set_mocap_quat('kuka_mocap', rotation)

        for _ in range(10):
            self.sim.step()

        # initial markers (index 3 is arbitrary)
        self.target_1 = self.sim.data.get_site_xpos('target_1')
        self.target_2 = self.sim.data.get_site_xpos('target_2')
        self.target_3 = self.sim.data.get_site_xpos('target_3')
        self.target_4 = self.sim.data.get_site_xpos('target_4')
        self.target_5 = self.sim.data.get_site_xpos('target_5')
        self.target_6 = self.sim.data.get_site_xpos('target_6')
        self.target_7 = self.sim.data.get_site_xpos('target_7')
        self.target_8 = self.sim.data.get_site_xpos('target_8')
        self.targets = [self.target_1, self.target_2, self.target_3, self.target_4, self.target_5, self.target_6,
                        self.target_7, self.target_8]

        self.init_center = self.sim.data.get_site_xpos('init_center')
        site_id = self.sim.model.site_name2id('init_center')
        sites_offset = (self.sim.data.site_xpos[site_id] - self.sim.model.site_pos[site_id]).copy()
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripper_tip').copy()

        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, -self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, -self.obj_range, 0.0]- sites_offset

        self.sim.step()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
