import pandas as pd
import matplotlib.pyplot as plt

# Read DFT results calculated in Q2(a)
data = pd.read_csv("dft_results.csv")

frequency = data["frequency"]
power = data["power"]

# Plot Power vs Frequency
plt.figure(figsize=(10, 6))

plt.plot(frequency, power)

plt.xlabel("Frequency")
plt.ylabel("Power")
plt.title("Power Spectrum of Annual River Flow")
plt.grid(True)

plt.tight_layout()

# Save the plot
plt.savefig("plots/Q2b_power_vs_frequency.png", dpi=300)

plt.show()