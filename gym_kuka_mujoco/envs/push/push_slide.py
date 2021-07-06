from gym_kuka_mujoco.envs.push import push_env
from gym import utils as gym_utils
from gym_kuka_mujoco.envs import utils, rotations
import numpy as np
MODEL_XML_PATH = 'R800_slide_gravity.xml'



class PushSlide(push_env.PushEnv, gym_utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'kuka_joint_1': 0.0,
            'kuka_joint_2': 0.942,
            'kuka_joint_3': 0.0,
            'kuka_joint_4': -1.32,
            'kuka_joint_5': 0.0,
            'kuka_joint_6': 0.879,
            'kuka_joint_7': 0
        }
        self.adapt_dict = dict()
        self.adapt_dict["field"] = [1.2, 0, 0.8, 0.625, 0.45, 0.2]
        self.adapt_dict["obstacles"] = []

        push_env.PushEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, n_substeps=20,
            target_in_the_air=False, target_offset=np.array([0.4, 0.0, 0.0]),
            obj_range=0.1, target_range=0.3, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type, n_actions=4)
        gym_utils.EzPickle.__init__(self)

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_needle_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
        else:
            goal = self.initial_needle_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_needle_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_needle_xpos[:2]) < 0.1:
                object_xpos = self.initial_needle_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)

        self.sim.forward()
        return True

    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('needle_tip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('needle_tip') * dt

        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # needle state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
            # TODO still need to implement!
        gripper_state = np.zeros(0)
        gripper_vel = np.zeros(0)
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        self.sim.forward()
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()

        target = self.sim.data.get_site_xpos('needle_tip').copy()
        self.rotation = self.sim.data.get_body_xquat('needle_tip').copy()
        self.sim.data.set_mocap_pos('kuka_mocap', target)
        self.sim.data.set_mocap_quat('kuka_mocap', self.rotation)
        for _ in range(10):
            self.sim.step()

        self.initial_needle_xpos = self.sim.data.get_site_xpos('needle_tip').copy()

        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]