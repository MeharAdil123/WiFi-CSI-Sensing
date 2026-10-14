import numpy as np
import matplotlib.pyplot as plt

# Load training data
X_train = np.load("./Data/MyCSI/data/x_train.npy")
y_train = np.load("./Data/MyCSI/label/y_train.npy")

class_names = [
    "Empty",
    "Sitting",
    "Standing",
    "Walking",
    "Falling"
]

# Calculate mean CSI amplitude for each sample
# X_train shape: (samples, 250, 90)
sample_mean = X_train.mean(axis=(1, 2))

# Collect mean amplitude for each class
class_data = []

for class_id in range(5):
    values = sample_mean[y_train == class_id]
    class_data.append(values)

# Print statistics
print("Mean CSI amplitude statistics:")
print("-" * 50)

for class_id, class_name in enumerate(class_names):
    values = class_data[class_id]

    print(
        f"{class_name:10s} | "
        f"Mean: {values.mean():.4f} | "
        f"Std: {values.std():.4f} | "
        f"Samples: {len(values)}"
    )

# Plot
plt.figure(figsize=(9, 6))

plt.boxplot(
    class_data,
    tick_labels=class_names
)

plt.xlabel("Activity")
plt.ylabel("Mean CSI Amplitude")
plt.title("CSI Amplitude Distribution Across Activities")

plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.show()