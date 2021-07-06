from gym import utils as gym_utils
from gym_kuka_mujoco.envs.fetch import fetch_env
from gym_kuka_mujoco.envs import rotations, utils

MODEL_XML_PATH = 'R800_pick_no_obstacle_gravity.xml'


class PickNoObstacleEnv(fetch_env.FetchEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'kuka_joint_1': 0.356,
            'kuka_joint_2': 0.942,
            'kuka_joint_3': 0.0,
            'kuka_joint_4': -1.53,
            'kuka_joint_5': 0.0,
            'kuka_joint_6': 0.628,
            'kuka_joint_7': 0.336,
            'r_gripper_finger_joint': 0.026,
            'l_gripper_finger_joint': 0.026
        }
        self.target_range_x = 0.2  # entire table: 0.125
        self.target_range_y = 0.10  # entire table: 0.175

        self.adapt_dict = dict()
        self.adapt_dict["field"] = [0.75, 0, 1.2, 0.25, 0.35, 0.2]
        self.adapt_dict["obstacles"] = []

        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, n_substeps=20,
            target_in_the_air=True, target_offset=0.0,
            obj_range=0.06, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        gym_utils.EzPickle.__init__(self)

    def _sample_goal(self):
        goal = self.target_center.copy()

        goal[1] += self.np_random.uniform(-self.target_range_y, self.target_range_y)
        goal[0] += self.np_random.uniform(-self.target_range_x, self.target_range_x)
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
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()[3]
        self.target_center = self.sim.data.get_site_xpos('target_center')
        self.init_center = self.sim.data.get_site_xpos('init_center')

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripper_tip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, -self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, self.obj_range, 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, -self.obj_range, 0.0] - sites_offset

        site_id = self.sim.model.site_name2id('mark1')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y,
                                                                 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark2')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y,
                                                                 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark3')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y,
                                                                 0.0] - sites_offset
        site_id = self.sim.model.site_name2id('mark4')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y,
                                                                 0.0] - sites_offset
        self.sim.step()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
