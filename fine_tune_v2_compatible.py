import numpy as np
from stable_baselines3 import PPO
from robot_tracking_env_v2_compatible import RobotTrackingEnvV2Compatible
from stable_baselines3.common.env_util import make_vec_env

# Create vectorized environment (same observation space as original)
env = make_vec_env(
    lambda: RobotTrackingEnvV2Compatible(
        model_path="franka_emika_panda/panda.xml",
        max_steps=500,
        freq=0.1,
        trajectory_type="figure8",
        control_delay=2,
        render_mode=None
    ),
    n_envs=4
)

# Load your existing model (21-dim observation)
model = PPO.load("ppo_franka_0.1hz_figure8_noise", env=env)

print("Fine-tuning on upgraded environment (exponential reward + control delay)...")
model.learn(total_timesteps=100_000)
model.save("ppo_franka_v2_compatible_final")

# Quick evaluation
eval_env = RobotTrackingEnvV2Compatible(
    model_path="franka_emika_panda/panda.xml",
    max_steps=500,
    freq=0.1,
    trajectory_type="figure8",
    control_delay=2,
    render_mode=None
)
obs, _ = eval_env.reset()
distances = []
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)
    distances.append(info["distance"])
    if truncated:
        break
print(f"Mean tracking error after fine-tuning: {np.mean(distances):.3f} m")
print(f"Final error: {distances[-1]:.3f} m")
eval_env.close()