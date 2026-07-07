import csv
import json
import time
from pathlib import Path

import numpy as np
from cnn_models import StandardCNN


DATA_PATH = "emnist_47_classes_16x16.npz"
FEEDBACK_DATA_PATH = "user_feedback_47_classes_16x16.npz"
MODEL_PATH = "emnist_47_brain.npz"

INPUT_DIM = 16
CONV_PADDING = 1
DENSE_HIDDEN_NODES = 64
EPOCHS = 7
LEARNING_RATE = 0.008
LR_DECAY_FACTOR = 0.5
LR_PLATEAU_PATIENCE = 1
MIN_LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-4
EARLY_STOPPING_MIN_EPOCHS = 7

MAX_SAMPLES = 23500
BALANCE_CLASSES = True
USE_FEEDBACK_DATA = True
TEST_RATIO = 0.1
VAL_RATIO = 0.1
RANDOM_SEED = 42
RESULTS_DIR = Path("results") / "baseline"


def decay_learning_rate(current_learning_rate):
    return max(MIN_LEARNING_RATE, current_learning_rate * LR_DECAY_FACTOR)


def resolve_data_path(path):
    local_path = Path(path)
    if local_path.exists():
        return local_path

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        matches = list(kaggle_input.rglob(local_path.name))
        if matches:
            return matches[0]
    return local_path


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


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
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    data_path = resolve_data_path(DATA_PATH)
    try:
        dataset = np.load(data_path)
        X_full = dataset["X"]
        y_full = dataset["y"]
    except FileNotFoundError:
        print("File not found:", data_path)
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

    base_train_counts = np.bincount(y_train, minlength=num_classes)
    val_counts = np.bincount(y_val, minlength=num_classes)
    test_counts = np.bincount(y_test, minlength=num_classes)
    feedback_samples = 0

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
            feedback_samples = len(X_feedback)

    final_train_counts = np.bincount(y_train, minlength=num_classes)
    split_rows = []
    for class_index, label in enumerate(unique_labels):
        split_rows.append({
            "class_index": class_index,
            "label": chr(int(label)),
            "unicode": int(label),
            "base_train_count": int(base_train_counts[class_index]),
            "feedback_count": int(final_train_counts[class_index] - base_train_counts[class_index]),
            "final_train_count": int(final_train_counts[class_index]),
            "validation_count": int(val_counts[class_index]),
            "test_count": int(test_counts[class_index]),
        })
    write_csv(
        RESULTS_DIR / "split_summary.csv",
        [
            "class_index", "label", "unicode", "base_train_count",
            "feedback_count", "final_train_count", "validation_count", "test_count",
        ],
        split_rows,
    )

    write_json(RESULTS_DIR / "run_config.json", {
        "data_path": str(data_path),
        "model_path": MODEL_PATH,
        "input_dim": INPUT_DIM,
        "num_classes": num_classes,
        "classes": [chr(int(label)) for label in unique_labels],
        "conv_padding": CONV_PADDING,
        "dense_hidden_nodes": DENSE_HIDDEN_NODES,
        "epochs": EPOCHS,
        "initial_learning_rate": LEARNING_RATE,
        "learning_rate_decay_factor": LR_DECAY_FACTOR,
        "learning_rate_scheduler": "reduce_on_validation_loss_plateau",
        "learning_rate_plateau_patience": LR_PLATEAU_PATIENCE,
        "minimum_learning_rate": MIN_LEARNING_RATE,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": EARLY_STOPPING_MIN_DELTA,
        "early_stopping_min_epochs": EARLY_STOPPING_MIN_EPOCHS,
        "max_samples": MAX_SAMPLES,
        "balance_classes": BALANCE_CLASSES,
        "use_feedback_data": USE_FEEDBACK_DATA,
        "feedback_samples": feedback_samples,
        "test_ratio": TEST_RATIO,
        "validation_ratio": VAL_RATIO,
        "random_seed": RANDOM_SEED,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "backend": "NumPy CPU",
    })

    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))
    print("Backend: NumPy CPU")

    model = StandardCNN(
        input_dim=INPUT_DIM,
        num_classes=num_classes,
        conv_padding=CONV_PADDING,
        hidden_nodes=DENSE_HIDDEN_NODES,
    )

    print("Starting training...")
    history = []
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    plateau_epochs = 0
    current_learning_rate = LEARNING_RATE
    learning_rate_reductions = 0
    stopped_early = False

    for epoch in range(EPOCHS):
        print("Epoch", epoch + 1, "Learning rate:", current_learning_rate)
        start_time = time.time()

        permutation = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]

        for i, (image, label) in enumerate(zip(X_train_shuffled, y_train_shuffled)):
            model.train_step(image, int(label), learning_rate=current_learning_rate)
            if (i + 1) % 500 == 0:
                print("Training image:", i + 1)

        print("Evaluating...")
        train_loss, train_acc = evaluate_model(model, X_train, y_train)
        val_loss, val_acc = evaluate_model(model, X_val, y_val)

        end_time = time.time()

        print("Time:", round(end_time - start_time, 2), "seconds")
        print("Train Loss:", round(train_loss, 4), "Train Acc:", round(train_acc, 2))
        print("Val Loss:", round(val_loss, 4), "Val Acc:", round(val_acc, 2))

        is_best = val_loss < (best_val_loss - EARLY_STOPPING_MIN_DELTA)
        if is_best:
            best_val_loss = float(val_loss)
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            plateau_epochs = 0
            model.save_weights(MODEL_PATH, class_labels=unique_labels)
            print("Saved new best checkpoint at epoch", best_epoch)
        else:
            epochs_without_improvement += 1
            plateau_epochs += 1
            print(
                "No validation improvement:",
                epochs_without_improvement,
                "/",
                EARLY_STOPPING_PATIENCE,
            )

        next_learning_rate = current_learning_rate
        learning_rate_reduced = False
        if (
            not is_best
            and plateau_epochs >= LR_PLATEAU_PATIENCE
            and current_learning_rate > MIN_LEARNING_RATE
        ):
            next_learning_rate = decay_learning_rate(current_learning_rate)
            learning_rate_reduced = next_learning_rate < current_learning_rate
            plateau_epochs = 0
            if learning_rate_reduced:
                learning_rate_reductions += 1
                print("Reducing learning rate to", next_learning_rate)

        history.append({
            "epoch": epoch + 1,
            "learning_rate": float(current_learning_rate),
            "next_learning_rate": float(next_learning_rate),
            "learning_rate_reduced": bool(learning_rate_reduced),
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "validation_loss": float(val_loss),
            "validation_accuracy": float(val_acc),
            "epoch_seconds": float(end_time - start_time),
            "is_best_checkpoint": bool(is_best),
        })
        write_csv(
            RESULTS_DIR / "training_history.csv",
            [
                "epoch", "learning_rate", "next_learning_rate",
                "learning_rate_reduced", "train_loss", "train_accuracy",
                "validation_loss", "validation_accuracy", "epoch_seconds",
                "is_best_checkpoint",
            ],
            history,
        )

        current_learning_rate = next_learning_rate

        if (
            epoch + 1 >= EARLY_STOPPING_MIN_EPOCHS
            and epochs_without_improvement >= EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            print("Early stopping triggered after epoch", epoch + 1)
            break

    if best_epoch == 0:
        raise RuntimeError("Training did not produce a finite validation checkpoint.")

    print("Loading best checkpoint from epoch", best_epoch)
    model.load_weights(MODEL_PATH)
    print("Testing best checkpoint...")
    test_loss, test_acc = evaluate_model(model, X_test, y_test)
    print("Test Loss:", round(test_loss, 4), "Test Acc:", round(test_acc, 2))
    write_json(RESULTS_DIR / "training_summary.json", {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "total_training_seconds": float(sum(row["epoch_seconds"] for row in history)),
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": float(history[best_epoch - 1]["validation_accuracy"]),
        "stopped_early": stopped_early,
        "final_learning_rate": float(current_learning_rate),
        "learning_rate_reductions": learning_rate_reductions,
        "dense_hidden_nodes": DENSE_HIDDEN_NODES,
        "model_parameters": int(sum(
            array.size for array in [
                model.layers[0].filters, model.layers[0].biases,
                model.layers[3].filters, model.layers[3].biases,
                model.layers[7].weights, model.layers[7].biases,
                model.layers[9].weights, model.layers[9].biases,
            ]
        )),
        "model_path": MODEL_PATH,
    })
    print("Saved training records to", RESULTS_DIR)


if __name__ == "__main__":
    main()
