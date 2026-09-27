import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# Given data
# --------------------------------------------------

T_data = np.array([0, 8, 16, 24, 32, 40], dtype=float)

O_data = np.array([
    14.621,
    11.843,
    9.870,
    8.418,
    7.305,
    6.413
], dtype=float)


# --------------------------------------------------
# Newton polynomial coefficients
# Obtained from Q1(a)
# --------------------------------------------------

a0 = 14.621
a1 = -0.34725
a2 = 0.0062890625
a3 = -9.244791666667e-05
a4 = 1.03759765625e-06
a5 = -9.663899739583e-09


# --------------------------------------------------
# Function for Newton polynomial
# --------------------------------------------------

def newton_polynomial(T):

    P = (
        a0
        + a1 * (T - 0)
        + a2 * (T - 0) * (T - 8)
        + a3 * (T - 0) * (T - 8) * (T - 16)
        + a4 * (T - 0) * (T - 8) * (T - 16) * (T - 24)
        + a5 * (T - 0) * (T - 8) * (T - 16) * (T - 24) * (T - 32)
    )

    return P


# --------------------------------------------------
# Quadratic spline coefficients
# Obtained from Q1(c)
# --------------------------------------------------

# S1: 0 <= T <= 8
a_1 = 14.621
b_1 = -0.34725
c_1 = 0.0

# S2: 8 <= T <= 16
a_2 = 11.843
b_2 = -0.34725
c_2 = 0.012578125

# S3: 16 <= T <= 24
a_3_s = 9.870
b_3 = -0.146
c_3 = -0.0044375

# S4: 24 <= T <= 32
a_4_s = 8.418
b_4 = -0.217
c_4 = 0.009734375

# S5: 32 <= T <= 40
a_5_s = 7.305
b_5 = -0.06125
c_5 = -0.00628125


# --------------------------------------------------
# Function for quadratic spline
# --------------------------------------------------

def quadratic_spline(T):

    S = np.zeros_like(T, dtype=float)

    # S1
    mask = (T >= 0) & (T < 8)

    x = T[mask] - 0

    S[mask] = (
        a_1
        + b_1 * x
        + c_1 * x**2
    )

    # S2
    mask = (T >= 8) & (T < 16)

    x = T[mask] - 8

    S[mask] = (
        a_2
        + b_2 * x
        + c_2 * x**2
    )

    # S3
    mask = (T >= 16) & (T < 24)

    x = T[mask] - 16

    S[mask] = (
        a_3_s
        + b_3 * x
        + c_3 * x**2
    )

    # S4
    mask = (T >= 24) & (T < 32)

    x = T[mask] - 24

    S[mask] = (
        a_4_s
        + b_4 * x
        + c_4 * x**2
    )

    # S5
    mask = (T >= 32) & (T <= 40)

    x = T[mask] - 32

    S[mask] = (
        a_5_s
        + b_5 * x
        + c_5 * x**2
    )

    return S


# --------------------------------------------------
# Generate values for plotting
# --------------------------------------------------

T_plot = np.linspace(0, 40, 500)

O_newton = newton_polynomial(T_plot)

O_spline = quadratic_spline(T_plot)


# --------------------------------------------------
# Plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    T_plot,
    O_newton,
    label="Newton 5th-order polynomial"
)

plt.plot(
    T_plot,
    O_spline,
    label="Quadratic spline"
)

plt.scatter(
    T_data,
    O_data,
    label="Given data points",
    zorder=5
)


# --------------------------------------------------
# Labels and formatting
# --------------------------------------------------

plt.xlabel("Temperature, T (°C)")
plt.ylabel("Dissolved Oxygen, O (mg/L)")

plt.title("Dissolved Oxygen vs Temperature")

plt.legend()

plt.grid(True)

plt.xlim(0, 40)

plt.tight_layout()


# --------------------------------------------------
# Save graph
# --------------------------------------------------

plt.savefig(
    "plots/Q1d_plot.png",
    dpi=300
)


# Show graph
plt.show()