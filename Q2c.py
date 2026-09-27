import pandas as pd

# Read DFT results from Q2(a)
data = pd.read_csv("dft_results.csv")

# Keep only non-zero positive frequencies up to Nyquist frequency
positive_data = data[
    (data["frequency"] > 0) &
    (data["frequency"] <= 0.5)
].copy()

# Sort according to power from highest to lowest
dominant = positive_data.sort_values(
    by="power",
    ascending=False
)

# Select the three most dominant frequencies
top_three = dominant.head(3)

print("Three Most Dominant Frequencies:")
print()

for _, row in top_three.iterrows():
    print(
        f"k = {int(row['k'])}, "
        f"frequency = {row['frequency']:.8f}, "
        f"power = {row['power']:.6e}"
    )