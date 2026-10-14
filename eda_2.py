# import numpy as np

# X_train = np.load("./Data/MyCSI/data/X_train.npy")
# X_val   = np.load("./Data/MyCSI/data/X_val.npy")
# X_test  = np.load("./Data/MyCSI/data/X_test.npy")

# y_train = np.load("./Data/MyCSI/label/y_train.npy")
# y_val   = np.load("./Data/MyCSI/label/y_val.npy")
# y_test  = np.load("./Data/MyCSI/label/y_test.npy")

# print("X_train:", X_train.shape)
# print("X_val:  ", X_val.shape)
# print("X_test: ", X_test.shape)

# print("y_train:", y_train.shape)
# print("y_val:  ", y_val.shape)
# print("y_test: ", y_test.shape)

# print("\nTotal samples:", len(X_train) + len(X_val) + len(X_test))



import numpy as np
import matplotlib.pyplot as plt

# Load data
X_train = np.load("./Data/MyCSI/data/X_train.npy")
X_val   = np.load("./Data/MyCSI/data/X_val.npy")
X_test  = np.load("./Data/MyCSI/data/X_test.npy")

y_train = np.load("./Data/MyCSI/label/y_train.npy")
y_val   = np.load("./Data/MyCSI/label/y_val.npy")
y_test  = np.load("./Data/MyCSI/label/y_test.npy")

# Combine the splits
X = np.concatenate([X_train, X_val, X_test], axis=0)
y = np.concatenate([y_train, y_val, y_test], axis=0)

class_names = [
    "Empty",
    "Sitting",
    "Standing",
    "Walking",
    "Falling"
]

# Create one heatmap for each activity
fig, axes = plt.subplots(5, 1, figsize=(10, 16))

for class_id, class_name in enumerate(class_names):

    # Find samples belonging to this activity
    indices = np.where(y == class_id)[0]

    # Take the first sample
    sample = X[indices[0]]

    # Remove unnecessary dimensions if present
    sample = np.squeeze(sample)

    axes[class_id].imshow(
        sample,
        aspect="auto",
        origin="upper"
    )

    axes[class_id].set_title(
        f"{class_name} — CSI Amplitude"
    )

    axes[class_id].set_ylabel("Packet / Time")

axes[-1].set_xlabel("CSI Feature (90)")

plt.tight_layout()

plt.savefig(
    "EDA_2_CSI_Amplitude_Heatmaps.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()