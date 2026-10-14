import numpy as np
import torch

from UT_HAR_model import UT_HAR_CNN_BiLSTM


MODEL_PATH = "checkpoints/cnn_bilstm_seed_456.pth"
TEST_DATA_PATH = "Data/MyCSI/data/X_test.npy"
TEST_LABEL_PATH = "Data/MyCSI/label/y_test.npy"

CLASS_NAMES = [
    "Empty",
    "Sitting",
    "Standing",
    "Walking",
    "Falling"
]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def main():
    print("Device:", device)

    # Load test data and labels
    X_test = np.load(TEST_DATA_PATH).astype(np.float32)
    y_test = np.load(TEST_LABEL_PATH).astype(np.int64)

    print("Test data shape:", X_test.shape)
    print("Test labels shape:", y_test.shape)

    # Match the normalization currently in dataset.py:
    # normalize the complete test split independently
    minimum = X_test.min()
    maximum = X_test.max()

    if maximum > minimum:
        X_test = (X_test - minimum) / (maximum - minimum)
    else:
        raise ValueError("Test dataset has no variation.")

    # Load model
    # model = UT_HAR_CNN_LSTM(num_classes=5)

    # state_dict = torch.load(
    #     MODEL_PATH,
    #     map_location=device,
    #     weights_only=True
    # )

    # model.load_state_dict(state_dict)

    model = UT_HAR_CNN_BiLSTM(num_classes=5)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    # Checkpoint contains metadata and the model weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)


    model.to(device)
    model.eval()

    # Predict every test sample
    X_tensor = torch.from_numpy(X_test).unsqueeze(1).to(device)

    with torch.inference_mode():
        logits = model(X_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predictions = probabilities.argmax(dim=1).cpu().numpy()

    accuracy = (predictions == y_test).mean()

    print(f"\nTest accuracy: {accuracy * 100:.2f}%")

    print("\nFirst 10 predictions:")

    for i in range(min(10, len(y_test))):
        predicted = CLASS_NAMES[int(predictions[i])]
        actual = CLASS_NAMES[int(y_test[i])]
        confidence = probabilities[i, predictions[i]].item() * 100

        print(
            f"Sample {i}: "
            f"Predicted={predicted}, "
            f"Actual={actual}, "
            f"Confidence={confidence:.2f}%"
        )


if __name__ == "__main__":
    main()