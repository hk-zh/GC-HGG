import os
import numpy as np
import gym
import copy
import mujoco_py
from gym.utils import seeding
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env
from mujoco_py.builder import MujocoException

from .assets import kuka_asset_dir
DEFAULT_SIZE_X = 1920
DEFAULT_SIZE_Y = 1080

class KukaGoalEnv(gym.GoalEnv):
    def __init__(self,
                 initial_qpos,
                 n_actions,
                 n_substeps,
                 model_path,
                 time_limit=3.,
                 timestep=0.002):
        full_path = os.path.join(kuka_asset_dir(), model_path)
        model = mujoco_py.load_model_from_path(full_path)
        model.opt.timestep = timestep
        self.sim = mujoco_py.MjSim(model, nsubsteps = n_substeps)
        self.viewer = None
        self._viewers = {}
        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }
        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())
        self.goal = self._sample_goal()
        obs = self._get_obs()
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), dtype='float32')
        self.observation_space = spaces.Dict(dict(
            desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
            observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        ))
        self.cameras = [[2.5, 90., -35.], [2.5, 135., -35], [2.5, 180., -35], [2.5, 225., -35], [2.5, 270, -35]]
        # distance, azimuth, elevation

        self.camera_pos = 0
        self.last_camera_pos = 0
        self.camera_num = len(self.cameras)
        self.last_action = None

    @property
    def dt(self):
        return self.sim.model.opt.timestep * self.sim.nsubsteps

    def seed(self, seed=None):
            self.np_random, seed = seeding.np_random(seed)
            return [seed]


    def step(self, action):
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info


    def reset(self):
        # Attempt to reset the simulator. Since we randomize initial conditions, it
        # is possible to get into a state with numerical issues (e.g. due to penetration or
        # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
        # In this case, we just keep randomizing until we eventually achieve a valid initial
        # configuration.
        super(KukaGoalEnv, self).reset()
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal().copy()
        obs = self._get_obs()
        return obs

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def render(self, mode='human', width=DEFAULT_SIZE_X, height=DEFAULT_SIZE_Y):
        self._render_callback()
        if mode == 'rgb_array':
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == 'human':
            self._get_viewer(mode).render()

    def set_camera_pos(self, pos: int):
        self.camera_pos = pos % self.camera_num

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == 'human':
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == 'rgb_array':
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        if self.camera_pos != self.last_camera_pos:
            self.last_camera_pos = self.camera_pos
            self._viewer_setup()
        return self.viewer



    def _env_setup(self, initial_qpos):
            """Initial configuration of the environment. Can be used to configure initial state
            and extract information from the simulation.
            """
            pass

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_obs(self):
        """Returns the observation.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal.
        """
        raise NotImplementedError()

    def _sample_goal(self):
        """Samples a new goal and returns it.
        """
        raise NotImplementedError()


    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass

    def _set_goal(self, goal):
        pass

