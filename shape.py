import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

# Newman et al. (1977) bottom-surface nodes
# https://www.cambridge.org/core/services/aop-cambridge-core/content/view/8A2761FBC15473F2CC76ADA9F9366AA9/S0022112025002058a.pdf/wake_transition_and_aerodynamics_of_a_dragonflyinspired_airfoil.pdf
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
    ]
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
    ]
)

# constant thickness
t = 0.005

# dense chordwise grid
x_dense = np.linspace(0, 1, 1000)

# Spline interpolation
y_bottom = CubicSpline(x_b, y_b, bc_type="natural")(x_dense)

y_top = y_bottom + t

# --------- Plot ----------
plt.figure(figsize=(10, 5))
plt.plot(x_dense, y_top, label="Top surface")
plt.plot(x_dense, y_bottom, label="Bottom surface")
plt.plot([x_dense[0], x_dense[0]], [y_top[0], y_bottom[0]], label="Leading edge")
plt.plot([x_dense[-1], x_dense[-1]], [y_top[-1], y_bottom[-1]], label="Trailing edge")
plt.xlim(-0.1, 1.05)
plt.ylim(-0.05, 0.10)
plt.title("Spline-smoothed corrugated wing section (Newman et al. model)")
plt.legend()
plt.tight_layout()
plt.show()
