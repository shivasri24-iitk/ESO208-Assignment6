import numpy as np
import pandas as pd


# --------------------------------------------------
# STEP 1: Read the Excel file
# --------------------------------------------------

file_path = "data/assignment6.xlsx"

data = pd.read_excel(file_path, engine="openpyxl")


# --------------------------------------------------
# STEP 2: Extract year and annual flow
# --------------------------------------------------

years = data["Year"].to_numpy()

flow = data["Annual flow (ac-ft/year)"].to_numpy(dtype=float)


# --------------------------------------------------
# STEP 3: Number of observations
# --------------------------------------------------

N = len(flow)

print("Number of observations =", N)
print("First year =", years[0])
print("Last year =", years[-1])


# --------------------------------------------------
# STEP 4: Calculate DFT manually
#
# X(k) = sum [ x(t) * exp(-i*2*pi*k*t/N) ]
#
# Here t = 0,1,2,...,N-1 in the code.
# This corresponds to:
#
# t = 1 -> year 1906
# t = 114 -> year 2019
# --------------------------------------------------

real_part = np.zeros(N)
imag_part = np.zeros(N)


for k in range(N):
    for t in range(1, N + 1):
        angle = (2 * np.pi * k * t) / N
        real_part[k] += flow[t - 1] * np.cos(angle)
        imag_part[k] -= flow[t - 1] * np.sin(angle)
# --------------------------------------------------
# STEP 5: Create complex Fourier coefficients
# --------------------------------------------------

fourier_coefficients = real_part + 1j * imag_part


# --------------------------------------------------
# STEP 6: Calculate frequency
# --------------------------------------------------

frequency = np.zeros(N)

for k in range(N):

    frequency[k] = k / N


# --------------------------------------------------
# STEP 7: Calculate power
# --------------------------------------------------

power = np.zeros(N)

for k in range(N):

    power[k] = (
        real_part[k] ** 2
        + imag_part[k] ** 2
    ) / (N ** 2)


# --------------------------------------------------
# STEP 8: Display Fourier coefficients
# --------------------------------------------------

print("\nFourier Transform Coefficients:\n")

for k in range(N):

    print(
        f"k = {k:3d}, "
        f"frequency = {frequency[k]:.8f}, "
        f"Real = {real_part[k]:.6f}, "
        f"Imaginary = {imag_part[k]:.6f}"
    )


# --------------------------------------------------
# STEP 9: Save results
# --------------------------------------------------

results = pd.DataFrame({

    "k": np.arange(N),

    "frequency": frequency,

    "real_part": real_part,

    "imag_part": imag_part,

    "power": power
})


results.to_csv(
    "dft_results.csv",
    index=False
)


print("\nDFT results saved successfully to dft_results.csv")