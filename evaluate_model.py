
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from UT_HAR_model import UT_HAR_CNN_LSTM

MODEL_PATH = "checkpoints/cnn_bilstm_seed_456.pth"
TEST_DATA_PATH = "Data/MyCSI/data/X_test.npy"
TEST_LABEL_PATH = "Data/MyCSI/label/y_test.npy"

CLASS_NAMES = ["Empty", "Sitting", "Standing", "Walking", "Falling"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    X_test = np.load(TEST_DATA_PATH).astype(np.float32)
    y_test = np.load(TEST_LABEL_PATH).astype(np.int64)

    minimum = X_test.min()
    maximum = X_test.max()
    if maximum <= minimum:
        raise ValueError("Test dataset has no variation.")

    X_test = (X_test - minimum) / (maximum - minimum)

    model = UT_HAR_CNN_LSTM(num_classes=5)
    checkpoint = torch.load(
        MODEL_PATH, map_location=device, weights_only=True
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    X_tensor = torch.from_numpy(X_test).unsqueeze(1).to(device)

    with torch.inference_mode():
        predictions = model(X_tensor).argmax(dim=1).cpu().numpy()

    print("Device:", device)
    print("Test samples:", len(y_test))
    print(f"Accuracy: {(predictions == y_test).mean() * 100:.2f}%")

    print("\nPer-class report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1, 2, 3, 4],
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1, 2, 3, 4]))


if __name__ == "__main__":
    main()
