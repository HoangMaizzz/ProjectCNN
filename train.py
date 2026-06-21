import time
import numpy as np
from cnn_models import StandardCNN


DATA_PATH = "emnist_47_classes_16x16.npz"
FEEDBACK_DATA_PATH = "user_feedback_47_classes_16x16.npz"
MODEL_PATH = "emnist_47_brain.npz"

INPUT_DIM = 16
CONV_PADDING = 1
EPOCHS = 10
LEARNING_RATE = 0.008

MAX_SAMPLES = 10000
BALANCE_CLASSES = True
USE_FEEDBACK_DATA = True
TEST_RATIO = 0.1
VAL_RATIO = 0.1
RANDOM_SEED = 42


def evaluate_model(model, X, y):
    loss = 0
    correct = 0
    num_samples = len(X)

    for image, label in zip(X, y):
        label = int(label)
        probabilities = model.forward(image)
        loss += -np.log(probabilities[label] + 1e-9)
        if np.argmax(probabilities) == label:
            correct += 1

    return (loss / num_samples), (correct / num_samples) * 100


def select_balanced_subset(X, y, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(X):
        if not BALANCE_CLASSES:
            return X, y

    rng = np.random.default_rng(seed)
    selected_indices = []
    unique_labels = np.unique(y)

    class_sizes = [np.sum(y == label) for label in unique_labels]
    samples_per_class = min(min(class_sizes), max(1, max_samples // len(unique_labels)))
    class_counts = {}

    for label in unique_labels:
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)
        take_count = min(samples_per_class, len(label_indices))
        selected_indices.extend(label_indices[:take_count])
        class_counts[label] = int(take_count)

    selected_indices = np.array(selected_indices)
    rng.shuffle(selected_indices)

    print("Selected class counts:", {chr(k): v for k, v in class_counts.items()})
    return X[selected_indices], y[selected_indices]


def select_subset(X, y, max_samples, balance_classes, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= len(X):
        if balance_classes:
            return select_balanced_subset(X, y, len(X), seed)
        return X, y

    if balance_classes:
        return select_balanced_subset(X, y, max_samples, seed)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(X))[:max_samples]
    return X[indices], y[indices]


def load_feedback_data(path, image_shape, valid_labels):
    try:
        feedback = np.load(path)
    except FileNotFoundError:
        return None, None

    X_feedback = feedback["X"]
    y_feedback = feedback["y"]
    if len(X_feedback) == 0:
        return None, None

    if X_feedback.ndim != 3 or X_feedback.shape[1:] != image_shape:
        print("Ignoring feedback: expected image shape", image_shape)
        return None, None

    valid_mask = np.isin(y_feedback, valid_labels)
    ignored_count = int(np.sum(~valid_mask))
    if ignored_count:
        print("Ignoring feedback samples with unsupported labels:", ignored_count)
        X_feedback = X_feedback[valid_mask]
        y_feedback = y_feedback[valid_mask]

    if len(X_feedback) == 0:
        return None, None

    print("Feedback samples:", len(X_feedback))
    return X_feedback, y_feedback


def stratified_split(X, y, test_ratio, val_ratio, seed):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        rng.shuffle(label_indices)

        num_test = int(round(len(label_indices) * test_ratio))
        num_val = int(round(len(label_indices) * val_ratio))

        test_indices.extend(label_indices[:num_test])
        val_indices.extend(label_indices[num_test:num_test + num_val])
        train_indices.extend(label_indices[num_test + num_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return (
        X[train_indices], X[val_indices], X[test_indices],
        y[train_indices], y[val_indices], y[test_indices],
    )


def encode_labels(y):
    unique_labels = np.unique(y)
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y_idx = np.array([label_to_index[label] for label in y])
    return y_idx, unique_labels


def main():
    print("Loading data...")
    try:
        dataset = np.load(DATA_PATH)
        X_full = dataset["X"]
        y_full = dataset["y"]
    except FileNotFoundError:
        print("File not found:", DATA_PATH)
        return

    X_full, y_full = select_subset(
        X_full,
        y_full,
        max_samples=MAX_SAMPLES,
        balance_classes=BALANCE_CLASSES,
        seed=RANDOM_SEED,
    )

    y_full_idx, unique_labels = encode_labels(y_full)
    num_classes = len(unique_labels)
    print("Classes:", [chr(label) for label in unique_labels])
    print("Dataset samples:", len(X_full))

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X_full,
        y_full_idx,
        test_ratio=TEST_RATIO,
        val_ratio=VAL_RATIO,
        seed=RANDOM_SEED,
    )

    if USE_FEEDBACK_DATA:
        X_feedback, y_feedback = load_feedback_data(
            FEEDBACK_DATA_PATH,
            image_shape=X_full.shape[1:],
            valid_labels=unique_labels,
        )
        if X_feedback is not None:
            y_feedback_idx = np.searchsorted(unique_labels, y_feedback)
            X_train = np.concatenate([X_train, X_feedback], axis=0)
            y_train = np.concatenate([y_train, y_feedback_idx], axis=0)

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
    print("Backend: NumPy CPU")

    model = StandardCNN(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        conv_padding=CONV_PADDING,
    )

    print("Starting training...")
    for epoch in range(EPOCHS):
        print("Epoch", epoch + 1)
        start_time = time.time()

        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i, (image, label) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
            model.train_step(image, int(label), learning_rate=LEARNING_RATE)
            if (i + 1) % 500 == 0:
                print("Training image:", i + 1)

        print("Evaluating...")
        train_loss, train_acc = evaluate_model(model, X_train, y_train)
        val_loss, val_acc = evaluate_model(model, X_val, y_val)

        end_time = time.time()

        print("Time:", round(end_time - start_time, 2), "seconds")
        print("Train Loss:", round(train_loss, 4), "Train Acc:", round(train_acc, 2))
        print("Val Loss:", round(val_loss, 4), "Val Acc:", round(val_acc, 2))

    print("Testing model...")
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    print("Test Loss:", round(test_loss, 4), "Test Acc:", round(test_acc, 2))
    model.save_weights(MODEL_PATH, class_labels=unique_labels)


if __name__ == "__main__":
    main()
