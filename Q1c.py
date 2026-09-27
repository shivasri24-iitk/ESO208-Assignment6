import numpy as np


# --------------------------------------------------
# Gaussian Elimination with Partial Pivoting
# --------------------------------------------------

def gauss_elimination(A, b):

    n = len(b)

    # Create augmented matrix
    augmented = np.column_stack((A.astype(float), b.astype(float)))

    # Forward elimination
    for k in range(n - 1):

        # Find pivot row
        pivot_row = k + np.argmax(np.abs(augmented[k:, k]))

        # Check for zero pivot
        if abs(augmented[pivot_row, k]) < 1e-12:
            raise ValueError("Singular matrix.")

        # Swap rows if required
        if pivot_row != k:
            augmented[[k, pivot_row]] = augmented[[pivot_row, k]]

        # Eliminate elements below pivot
        for i in range(k + 1, n):

            factor = augmented[i, k] / augmented[k, k]

            augmented[i, k:] = (
                augmented[i, k:]
                - factor * augmented[k, k:]
            )

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = (
            augmented[i, -1]
            - np.dot(
                augmented[i, i + 1:n],
                x[i + 1:n]
            )
        ) / augmented[i, i]

    return x


# --------------------------------------------------
# Unknowns:
#
# a1, b1, c1,
# a2, b2, c2,
# a3, b3, c3,
# a4, b4, c4,
# a5, b5, c5
# --------------------------------------------------

A = np.zeros((15, 15))
b = np.zeros(15)

row = 0


# --------------------------------------------------
# Function-value equations
# --------------------------------------------------

# a1 = 14.621
A[row, 0] = 1
b[row] = 14.621
row += 1


# a1 + 8b1 + 64c1 = 11.843
A[row, 0] = 1
A[row, 1] = 8
A[row, 2] = 64
b[row] = 11.843
row += 1


# a2 = 11.843
A[row, 3] = 1
b[row] = 11.843
row += 1


# a2 + 8b2 + 64c2 = 9.870
A[row, 3] = 1
A[row, 4] = 8
A[row, 5] = 64
b[row] = 9.870
row += 1


# a3 = 9.870
A[row, 6] = 1
b[row] = 9.870
row += 1


# a3 + 8b3 + 64c3 = 8.418
A[row, 6] = 1
A[row, 7] = 8
A[row, 8] = 64
b[row] = 8.418
row += 1


# a4 = 8.418
A[row, 9] = 1
b[row] = 8.418
row += 1


# a4 + 8b4 + 64c4 = 7.305
A[row, 9] = 1
A[row, 10] = 8
A[row, 11] = 64
b[row] = 7.305
row += 1


# a5 = 7.305
A[row, 12] = 1
b[row] = 7.305
row += 1


# a5 + 8b5 + 64c5 = 6.413
A[row, 12] = 1
A[row, 13] = 8
A[row, 14] = 64
b[row] = 6.413
row += 1


# --------------------------------------------------
# First derivative continuity
# --------------------------------------------------

# b1 + 16c1 - b2 = 0
A[row, 1] = 1
A[row, 2] = 16
A[row, 4] = -1
row += 1


# b2 + 16c2 - b3 = 0
A[row, 4] = 1
A[row, 5] = 16
A[row, 7] = -1
row += 1


# b3 + 16c3 - b4 = 0
A[row, 7] = 1
A[row, 8] = 16
A[row, 10] = -1
row += 1


# b4 + 16c4 - b5 = 0
A[row, 10] = 1
A[row, 11] = 16
A[row, 13] = -1
row += 1


# --------------------------------------------------
# Boundary condition
# c1 = 0
# --------------------------------------------------

A[row, 2] = 1
b[row] = 0


# --------------------------------------------------
# Solve the system
# --------------------------------------------------

solution = gauss_elimination(A, b)


# --------------------------------------------------
# Display coefficients
# --------------------------------------------------

print("Quadratic Spline Coefficients:\n")

for i in range(5):

    a = solution[3 * i]
    bb = solution[3 * i + 1]
    c = solution[3 * i + 2]

    print(
        f"S{i + 1}: "
        f"a = {a:.12f}, "
        f"b = {bb:.12f}, "
        f"c = {c:.12f}"
    )