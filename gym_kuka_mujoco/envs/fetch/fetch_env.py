import numpy as np
from gym_kuka_mujoco.envs import utils, rotations
from gym_kuka_mujoco.envs import kuka_goal_env


def goal_distance(goal_a: np.ndarray, goal_b: np.ndarray) -> float:
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchEnv(kuka_goal_env.KukaGoalEnv):
    """
    Superclass for all Kuka fetch environments
    """

    def __init__(
            self, model_path, n_substeps,
            has_object, target_in_the_air, target_offset, obj_range, target_range,
            distance_threshold, initial_qpos, reward_type,
    ):
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        super(FetchEnv, self).__init__(
            model_path=model_path, n_substeps=n_substeps, n_actions=4,
            initial_qpos=initial_qpos
        )
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripper_tip').copy()

        self.cameras = [[2, 135., -12]]
        self.camera_pos = 0
        # GoalEnv methods
        # ----------------------------

    def compute_reward(self, achieved_goal: np.ndarray, desired_goal: np.ndarray, info):
        d = goal_distance(achieved_goal, desired_goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def _step_callback(self):
        pass

    def _set_action(self, action):
        assert action.shape == (4,)
        action = action.copy()
        pos_ctrl, gripper_ctrl = action[:3], action[3]
        pos_ctrl *= 0.05
        rot_ctrl = [0., 0., 1. , 0.]
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        utils.gripper_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)



    def _get_obs(self):
        grip_pos = self.sim.data.get_site_xpos('gripper_tip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('gripper_tip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
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
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:]
        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }

    def _viewer_setup(self):
        '''
        set viewer to camera position
        '''
        body_id = self.sim.model.body_name2id('gripper_entity')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = self.cameras[self.camera_pos][0]
        self.viewer.cam.azimuth = self.cameras[self.camera_pos][1]
        self.viewer.cam.elevation = self.cameras[self.camera_pos][2]

    def _render_callback(self):
        # Visualize target.
        site_id = self.sim.model.site_name2id('init_center')
        sites_offset = (self.sim.data.site_xpos[site_id] - self.sim.model.site_pos[site_id]).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset
        self.sim.forward()


    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range, self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos
            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            # TODO still need to implement!

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
        return goal.copy()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)



    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        target = self.sim.data.get_site_xpos('gripper_tip').copy()
        rotation = self.sim.data.get_body_xquat('gripper_tip').copy()
        self.sim.data.set_mocap_pos('kuka_mocap', target)
        self.sim.data.set_mocap_quat('kuka_mocap', rotation)
        for _ in range(50):
            self.sim.step()

        self.initial_gripper_xpos = self.sim.data.get_site_xpos('gripper_tip').copy()
        if self.has_object:
            self.height_offset = self.sim.data.get_site_xpos('object0')[2]


