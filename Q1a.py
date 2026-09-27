import numpy as np

# Given data
T = np.array([0, 8, 16, 24, 32, 40], dtype=float)
O = np.array([14.621, 11.843, 9.870, 8.418, 7.305, 6.413], dtype=float)

# Number of data points
n = len(T)

# Create divided difference table
DD = np.zeros((n, n))

# First column contains the given O values
DD[:, 0] = O

# Calculate divided differences
for j in range(1, n):
    for i in range(n - j):
        DD[i, j] = (
            DD[i + 1, j - 1] - DD[i, j - 1]
        ) / (T[i + j] - T[i])

# Newton polynomial coefficients
coefficients = DD[0, :]

# Display divided difference table
print("Divided Difference Table:")
print(DD)

# Display Newton coefficients
print("\nNewton Polynomial Coefficients:")

for i in range(n):
    print(f"a{i} = {coefficients[i]:.12e}")

# Display Newton polynomial
print("\nNewton Interpolating Polynomial:")

print(f"P(T) = {coefficients[0]:.12f}", end="")

for i in range(1, n):
    print(f" + ({coefficients[i]:.12e})", end="")

    for j in range(i):
        print(f"(T - {T[j]:g})", end="")

print()