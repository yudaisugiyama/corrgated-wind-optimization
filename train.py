import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise

from env import WingOptimizationEnv

# ----------------- 1) Make env ------------------------------------------- #
ENV_STEPS = 30
env = WingOptimizationEnv(iter_step=ENV_STEPS)

# ----------------- 2) Action noise (exploration) ------------------------- #
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

# ----------------- 3) TD3 agent ------------------------------------------ #
model = TD3(
    policy="MlpPolicy",
    env=env,
    action_noise=action_noise,
    learning_rate=1e-3,
    buffer_size=200_000,
    learning_starts=1_000,
    batch_size=256,
    gamma=0.99,
    tau=0.005,
    policy_delay=2,
    verbose=1,
)

# ----------------- 4) Train ---------------------------------------------- #
model.learn(total_timesteps=5_000)
model.save("models/td3_corrugated")

print("Training finished. Model saved to models/td3_corrugated")
