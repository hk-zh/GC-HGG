from gym_kuka_mujoco.envs.push import push_env
from gym import utils as gym_utils
from gym_kuka_mujoco.envs import utils, rotations
MODEL_XML_PATH = 'R800_push_labyrinth_gravity_v2.xml'


class PushLabyrinthEnvV2(push_env.PushEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'kuka_joint_1': 0.326,
            'kuka_joint_2': 1.07,
            'kuka_joint_3': 0.0,
            'kuka_joint_4': -1.74,
            'kuka_joint_5': 0.0,
            'kuka_joint_6': 0.335,
            'kuka_joint_7': 0.275
        }
        self.target_range_x = 0.05
        self.target_range_y = 0.05

        self.adapt_dict = dict()
        self.adapt_dict["field"] = [0.75, 0, 1.05-0.15, 0.25, 0.35, 0.05]
        self.adapt_dict["obstacles"] = [[0.65, 0, 1.1-0.15, 0.11, 0.02, 0.1], [0.52, 0, 1.1-0.15, 0.02, 0.35, 0.1], [0.78, 0, 1.1-0.15, 0.02, 0.2, 0.1]]

        push_env.PushEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, n_substeps=20,
            target_in_the_air=False, target_offset=0.0,
            obj_range=0.05, target_range=0.05, distance_threshold=0.05,
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
        target = self.sim.data.get_site_xpos('needle_tip').copy()
        self.rotation = self.sim.data.get_body_xquat('needle_tip').copy()
        self.sim.data.set_mocap_pos('kuka_mocap', target)
        self.sim.data.set_mocap_quat('kuka_mocap', self.rotation)

        for _ in range(10):
            self.sim.step()
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()[3]
        self.target_center = self.sim.data.get_site_xpos('target_center')
        self.init_center = self.sim.data.get_site_xpos('init_center')

        self.initial_needle_xpos = self.sim.data.get_site_xpos('needle_tip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

        site_id = self.sim.model.site_name2id('init_1')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_2')
        self.sim.model.site_pos[site_id] = self.init_center + [self.obj_range, -self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_3')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, self.obj_range, 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('init_4')
        self.sim.model.site_pos[site_id] = self.init_center + [-self.obj_range, -self.obj_range, 0.0]- sites_offset

        site_id = self.sim.model.site_name2id('mark1')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, self.target_range_y,
                                                                 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('mark2')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, self.target_range_y,
                                                                 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('mark3')
        self.sim.model.site_pos[site_id] = self.target_center + [self.target_range_x, -self.target_range_y,
                                                                 0.0]- sites_offset
        site_id = self.sim.model.site_name2id('mark4')
        self.sim.model.site_pos[site_id] = self.target_center + [-self.target_range_x, -self.target_range_y,
                                                                 0.0]- sites_offset
        self.sim.step()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]
