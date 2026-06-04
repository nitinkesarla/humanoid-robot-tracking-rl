import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from stable_baselines3 import PPO
from robot_tracking_env_upgraded import RobotTrackingEnvUpgraded

# Load the final model
model = PPO.load("ppo_franka_v2_compatible_final")

# Environment settings
env = RobotTrackingEnvUpgraded(
    model_path="franka_emika_panda/panda.xml",
    max_steps=500,
    freq=0.1,
    trajectory_type="figure8",
    render_mode=None
)

# Evaluate multiple episodes
n_episodes = 5
all_errors = []
all_trajectories_target = []
all_trajectories_ee = []

for ep in range(n_episodes):
    obs, _ = env.reset()
    distances = []
    target_positions = []
    ee_positions = []
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        distances.append(info["distance"])
        # Capture positions for 3D plot (sample every 10 steps to keep size manageable)
        if step % 10 == 0:
            t = step * env.dt
            target_positions.append(env._get_desired_pose(t))
            ee_positions.append(env._get_ee_position())
        if truncated:
            break
    all_errors.append(distances)
    all_trajectories_target.append(np.array(target_positions))
    all_trajectories_ee.append(np.array(ee_positions))

# ---- 1. Tracking error over time (multiple episodes) ----
plt.figure(figsize=(12, 5))
for i, errors in enumerate(all_errors):
    plt.plot(errors, label=f'Episode {i+1}', alpha=0.7)
plt.xlabel('Step')
plt.ylabel('Tracking Error (m)')
plt.title('End-Effector Tracking Error Over Time (PPO, Figure‑8, Noise)')
plt.legend()
plt.grid(True)
plt.savefig('tracking_error_multiple_episodes.png', dpi=150)
print("Saved tracking_error_multiple_episodes.png")

# ---- 2. 3D trajectory: target vs end-effector (first episode) ----
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
target_traj = all_trajectories_target[0]
ee_traj = all_trajectories_ee[0]
ax.plot(target_traj[:, 0], target_traj[:, 1], target_traj[:, 2], 'b-', linewidth=2, label='Target')
ax.plot(ee_traj[:, 0], ee_traj[:, 1], ee_traj[:, 2], 'r--', linewidth=2, label='End-Effector')
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Trajectory: Target vs End-Effector')
ax.legend()
plt.savefig('trajectory_3d.png', dpi=150)
print("Saved trajectory_3d.png")

# ---- 3. Cartesian components (X, Y, Z over time) ----
t = np.arange(0, len(target_traj)) * (env.dt * 10)  # because we sampled every 10 steps
plt.figure(figsize=(12, 10))
components = ['X', 'Y', 'Z']
for i, comp in enumerate(components):
    plt.subplot(3, 1, i+1)
    plt.plot(t, target_traj[:, i], 'b-', label=f'Target {comp}')
    plt.plot(t, ee_traj[:, i], 'r--', label=f'EE {comp}')
    plt.ylabel(f'{comp} (m)')
    plt.legend()
    plt.grid(True)
plt.xlabel('Time (s)')
plt.suptitle('Cartesian Components: Target vs End-Effector')
plt.tight_layout()
plt.savefig('cartesian_components.png', dpi=150)
print("Saved cartesian_components.png")

# ---- 4. Error distribution across all episodes ----
all_errors_flat = np.concatenate(all_errors)
plt.figure(figsize=(10, 5))
plt.hist(all_errors_flat, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(np.mean(all_errors_flat), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(all_errors_flat):.4f} m')
plt.axvline(np.median(all_errors_flat), color='green', linestyle='dashed', linewidth=2, label=f'Median: {np.median(all_errors_flat):.4f} m')
plt.xlabel('Tracking Error (m)')
plt.ylabel('Frequency')
plt.title('Error Distribution Across All Episodes')
plt.legend()
plt.grid(True)
plt.savefig('error_distribution.png', dpi=150)
print("Saved error_distribution.png")

# Print summary statistics
print("\nSummary Statistics ({} evaluation episodes):".format(n_episodes))
print(f"  Mean tracking error:   {np.mean(all_errors_flat):.4f} m")
print(f"  Median tracking error: {np.median(all_errors_flat):.4f} m")
print(f"  Std tracking error:    {np.std(all_errors_flat):.4f} m")
print(f"  90th percentile error: {np.percentile(all_errors_flat, 90):.4f} m")
print(f"  Max error:             {np.max(all_errors_flat):.4f} m")

print("\nAll plots saved. You can now include them in your submission.")