import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import TD3

from env import X_DENSE, WingOptimizationEnv, build_airfoil_shape

# ----------------- 1) Load env & model ----------------------------------- #
env = WingOptimizationEnv(iter_step=30)
model = TD3.load("models/td3_corrugated")

# ----------------- 2) Rollout ------------------------------------------- #
best_score = -np.inf
best_params = None

N_EPISODES = 100
for _ in range(N_EPISODES):
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
    if reward > best_score:
        best_score = reward
        best_params = obs.copy()

assert best_params is not None, "No episode completed?"

# ----------------- 3) Save result --------------------------------------- #
Path("results").mkdir(exist_ok=True)
y_top, y_bottom = build_airfoil_shape(best_params)
out = {
    "params": best_params.tolist(),
    "cl_cd_proxy": float(best_score),
    "x": X_DENSE.tolist(),
    "y_top": y_top.tolist(),
    "y_bottom": y_bottom.tolist(),
}

with open("results/best_shape.json", "w") as f:
    json.dump(out, f, indent=2)

print(f"Best proxy CL/CD  : {best_score:.3f}")
print("Result saved to  : results/best_shape.json")

# ----------------- 4) Visualize final shape ----------------------------- #

plt.figure(figsize=(12, 6))
plt.plot(X_DENSE, y_top, "b-", label="Top surface", linewidth=2)
plt.plot(X_DENSE, y_bottom, "r-", label="Bottom surface", linewidth=2)
plt.plot([X_DENSE[0], X_DENSE[0]], [y_top[0], y_bottom[0]], "k-", label="Leading edge", linewidth=2)
plt.plot([X_DENSE[-1], X_DENSE[-1]], [y_top[-1], y_bottom[-1]], "k-", label="Trailing edge", linewidth=2)
plt.fill_between(X_DENSE, y_top, y_bottom, alpha=0.3, color="gray")
plt.xlim(-0.05, 1.05)
plt.ylim(min(y_bottom.min(), y_top.min()) - 0.01, max(y_bottom.max(), y_top.max()) + 0.01)
plt.xlabel("Chord position")
plt.ylabel("Height")
plt.title(f"Optimized Corrugated Wing Shape (CL/CD proxy: {best_score:.3f})")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
