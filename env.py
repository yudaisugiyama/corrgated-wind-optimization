from typing import Any

import gym
import numpy as np
from gym import spaces
from scipy.interpolate import CubicSpline

# --------------------------------------------------------------------- #
# ★ 1) 形状ベースライン（コルゲート翼断面）
# --------------------------------------------------------------------- #
x_b = np.array(
    [
        0.00,
        0.09,
        0.14,
        0.19,
        0.24,
        0.29,
        0.34,
        0.50,
        0.60,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
        0.95,
        1.00,
    ],
    dtype=np.float32,
)
y_b = np.array(
    [
        0.000,
        0.000,
        0.055,
        0.000,
        0.055,
        0.000,
        0.000,
        0.020,
        0.060,
        0.080,
        0.085,
        0.080,
        0.070,
        0.055,
        0.040,
        0.025,
    ],
    dtype=np.float32,
)
T_THICKNESS = 0.005  # constant thickness
X_DENSE = np.linspace(0.0, 1.0, 1_000)  # chordwise grid (fixed)


# --------------------------------------------------------------------- #
# 2) 幾何関数
# --------------------------------------------------------------------- #
def build_airfoil_shape(params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    params : (n_cp,) Additive offsets in *normalized* range [-1,1].
    return : (y_top, y_bottom) at X_DENSE
    """
    scale = 0.02  # 1.0 → 20 mm offset (例)
    y_control = y_b + scale * params
    y_bottom = CubicSpline(x_b, y_control, bc_type="natural")(X_DENSE)
    y_top = y_bottom + T_THICKNESS
    return y_top, y_bottom


def compute_lift_drag(params: np.ndarray) -> float:
    """
    Dummy CL/CD evaluator. Replace with real CFD solver.
    """
    # TODO: Replace with FreeFEM++ call.
    """
    擬似報酬: 断面の最大厚みが目標値 *T_THICKNESS* に近いほど高得点。
    ------------------------------------------------------------------
    ▸ diff      : y_top.max() − y_bottom.min()
    ▸ error     : |diff − T_THICKNESS|
    ▸ reward    : 1 / (1 + error)   （= 1 が最良、エラーが大きいほど小さくなる）
    ------------------------------------------------------------------
    """
    y_top, y_bottom = build_airfoil_shape(params)
    diff = float(y_top.max() - y_bottom.min())
    error = abs(diff - T_THICKNESS)
    return 1 / (1 + error**2)


# --------------------------------------------------------------------- #
# 3) Gym 環境クラス
# --------------------------------------------------------------------- #
class WingOptimizationEnv(gym.Env):
    """
    Corrugated wing CL/CD maximization.
    Each action tweaks control-point offsets; reward = pseudo CL/CD.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        iter_step: int = 30,
        param_bound: float = 1.0,
    ):
        super().__init__()
        self.n_params: int = len(x_b)  # 16 control points
        self.iter_step = iter_step
        self.param_bound = param_bound

        # Action / Observation space
        self.action_space = spaces.Box(low=-0.05, high=0.05, shape=(self.n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-param_bound, high=param_bound, shape=(self.n_params,), dtype=np.float32
        )

        # Internal state
        self.params: np.ndarray | None = None
        self.counter: int = 0

    # ----------------------------------------------------------------- #
    def reset(self) -> np.ndarray:
        self.params = self.np_random.uniform(low=-0.2, high=0.2, size=self.n_params).astype(np.float32)
        self.counter = 0
        return self.params.copy()

    # ----------------------------------------------------------------- #
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        self.params = np.clip(self.params + action.astype(np.float32), -self.param_bound, self.param_bound)

        reward = compute_lift_drag(self.params)
        self.counter += 1
        done = self.counter >= self.iter_step

        info = {"step": self.counter, "cl_cd": reward}
        return self.params.copy(), reward, done, info

    # ----------------------------------------------------------------- #
    def render(self, mode: str = "human") -> None:
        y_top, y_bottom = build_airfoil_shape(self.params)
        print(f"step {self.counter:02d} | CL/CD proxy ≈ {compute_lift_drag(self.params):.3f}")

    def close(self) -> None:
        pass
