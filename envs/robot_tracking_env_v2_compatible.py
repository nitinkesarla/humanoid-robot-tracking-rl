import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
from typing import Optional, Tuple
from collections import deque

class RobotTrackingEnvV2Compatible(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        model_path: str = "franka_emika_panda/panda.xml",
        max_steps: int = 500,
        render_mode: Optional[str] = None,
        freq: float = 0.1,
        trajectory_type: str = "figure8",
        control_delay: int = 2,
    ):
        super().__init__()
        self.model_path = model_path
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.freq = freq
        self.trajectory_type = trajectory_type
        self.control_delay = control_delay

        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.dt = self.model.opt.timestep

        # Action space: target joint positions (7 joints)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)

        # Observation space: same as original (21 dims)
        # joint positions (7) + joint velocities (7) + desired EE pos (3) + current EE pos (3) + time (1) = 21
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32)

        # End‑effector body id (hand)
        self.ee_body_id = 9

        # Joint position limits (Franka)
        self.joint_low = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.joint_high = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

        # Trajectory parameters
        self.center = np.array([0.5, 0.0, 0.5])
        self.radius = 0.15

        # Delay buffer
        self.action_buffer = deque(maxlen=control_delay)
        for _ in range(control_delay):
            self.action_buffer.append(np.zeros(7))

        self.prev_action = np.zeros(7)
        self.step_count = 0
        self.prev_distance = None

    def _get_ee_position(self) -> np.ndarray:
        return self.data.xpos[self.ee_body_id].copy()

    def _get_desired_pose(self, t: float) -> np.ndarray:
        angle = 2 * np.pi * self.freq * t
        if self.trajectory_type == "circle":
            dx = self.radius * np.cos(angle)
            dy = self.radius * np.sin(angle)
        else:  # figure8
            dx = self.radius * np.sin(angle)
            dy = self.radius * np.sin(angle) * np.cos(angle)
        return self.center + np.array([dx, dy, 0.0])

    def _get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    def _get_joint_velocities(self) -> np.ndarray:
        return self.data.qvel[:7].copy()

    def _get_observation(self, t: float) -> np.ndarray:
        ee_pos = self._get_ee_position()
        desired_pos = self._get_desired_pose(t)
        obs = np.concatenate([
            self._get_joint_positions(),
            self._get_joint_velocities(),
            desired_pos,
            ee_pos,
            [t]
        ])
        # Add observation noise (as before)
        noise = np.random.normal(0, 0.01, size=obs.shape)
        obs = obs + noise
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self.prev_action = np.zeros(7)
        self.prev_distance = None
        self.action_buffer.clear()
        for _ in range(self.control_delay):
            self.action_buffer.append(np.zeros(7))
        return self._get_observation(0.0), {}

    def step(self, action: np.ndarray):
        # Apply delayed action (oldest in buffer)
        delayed_action = self.action_buffer.popleft()
        self.action_buffer.append(action.copy())

        # Position control
        target_pos = self.joint_low + (delayed_action + 1) / 2 * (self.joint_high - self.joint_low)
        self.data.ctrl[:7] = target_pos
        mujoco.mj_step(self.model, self.data)

        t = self.step_count * self.dt
        self.step_count += 1

        ee_pos = self._get_ee_position()
        desired_pos = self._get_desired_pose(t)
        distance = np.linalg.norm(desired_pos - ee_pos)

        # Exponential reward (much stronger gradient near zero)
        reward = np.exp(-5 * distance)

        # Bonus for being extremely close
        if distance < 0.05:
            reward += 1.0

        # Smoothness penalty
        reward -= 0.001 * np.linalg.norm(self._get_joint_velocities())

        # Improvement bonus
        if self.prev_distance is not None and distance < self.prev_distance:
            reward += 0.1
        self.prev_distance = distance

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_observation(t)
        info = {"distance": distance}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                try:
                    from mujoco import viewer
                    self.viewer = viewer.launch_passive(self.model, self.data)
                except:
                    print("Rendering not available.")
                    self.render_mode = None
            if self.viewer:
                self.viewer.sync()

    def close(self):
        if self.viewer:
            self.viewer.close()