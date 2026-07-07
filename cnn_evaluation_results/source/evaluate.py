import csv
import json
import time
import zipfile
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    matplotlib = None
    plt = None

import train
from cnn_models import StandardCNN


OUTPUT_DIR = Path("results") / "baseline" / "evaluation"
BASELINE_DIR = Path("results") / "baseline"
ARCHIVE_PATH = Path("cnn_evaluation_results.zip")
MAX_ERROR_IMAGES = 25
INFERENCE_WARMUP_IMAGES = 10
TRAINING_HISTORY_COLUMNS = {
    "epoch", "learning_rate", "next_learning_rate", "learning_rate_reduced",
    "train_loss", "train_accuracy", "validation_loss", "validation_accuracy",
    "epoch_seconds", "is_best_checkpoint",
}
TRAINING_SUMMARY_KEYS = {
    "test_loss", "test_accuracy", "total_training_seconds", "completed_epochs",
    "best_epoch", "best_validation_loss", "best_validation_accuracy",
    "stopped_early", "final_learning_rate", "learning_rate_reductions",
    "dense_hidden_nodes", "model_parameters", "model_path",
}


def require_matplotlib():
    if plt is None:
        raise SystemExit(
            "Evaluation charts require matplotlib. Install it with: pip install matplotlib"
        )


def prepare_data_splits():
    dataset = np.load(train.resolve_data_path(train.DATA_PATH), allow_pickle=False)
    X, y = train.select_subset(
        dataset["X"],
        dataset["y"],
        max_samples=train.MAX_SAMPLES,
        balance_classes=train.BALANCE_CLASSES,
        seed=train.RANDOM_SEED,
    )
    y_encoded, class_labels = train.encode_labels(y)
    X_train, X_val, X_test, y_train, y_val, y_test = train.stratified_split(
        X,
        y_encoded,
        test_ratio=train.TEST_RATIO,
        val_ratio=train.VAL_RATIO,
        seed=train.RANDOM_SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test, class_labels


def prepare_test_data():
    _, _, X_test, _, _, y_test, class_labels = prepare_data_splits()
    return X_test, y_test, class_labels


def remove_stale_training_artifacts():
    history_path = BASELINE_DIR / "training_history.csv"
    summary_path = BASELINE_DIR / "training_summary.json"
    curve_path = OUTPUT_DIR / "training_curves.png"

    stale = False
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        stale = not rows or not TRAINING_HISTORY_COLUMNS.issubset(rows[0].keys())

    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        stale = stale or not TRAINING_SUMMARY_KEYS.issubset(summary.keys())
        stale = stale or summary.get("dense_hidden_nodes") != train.DENSE_HIDDEN_NODES

    if stale:
        print("Removing stale baseline training records from an older run.")
        for path in [history_path, summary_path, curve_path]:
            if path.exists():
                path.unlink()


def sync_baseline_metadata(X_train, X_val, X_test, y_train, y_val, y_test, class_labels):
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    num_classes = len(class_labels)
    train_counts = np.bincount(y_train, minlength=num_classes)
    val_counts = np.bincount(y_val, minlength=num_classes)
    test_counts = np.bincount(y_test, minlength=num_classes)

    split_rows = []
    for class_index, label in enumerate(class_labels):
        split_rows.append({
            "class_index": class_index,
            "label": chr(int(label)),
            "unicode": int(label),
            "base_train_count": int(train_counts[class_index]),
            "feedback_count": 0,
            "final_train_count": int(train_counts[class_index]),
            "validation_count": int(val_counts[class_index]),
            "test_count": int(test_counts[class_index]),
        })
    write_csv(
        BASELINE_DIR / "split_summary.csv",
        [
            "class_index", "label", "unicode", "base_train_count",
            "feedback_count", "final_train_count", "validation_count", "test_count",
        ],
        split_rows,
    )

    write_json(BASELINE_DIR / "run_config.json", {
        "data_path": str(train.resolve_data_path(train.DATA_PATH)),
        "model_path": train.MODEL_PATH,
        "input_dim": train.INPUT_DIM,
        "num_classes": num_classes,
        "classes": [chr(int(label)) for label in class_labels],
        "conv_padding": train.CONV_PADDING,
        "dense_hidden_nodes": train.DENSE_HIDDEN_NODES,
        "epochs": train.EPOCHS,
        "initial_learning_rate": train.LEARNING_RATE,
        "learning_rate_decay_factor": train.LR_DECAY_FACTOR,
        "learning_rate_scheduler": "reduce_on_validation_loss_plateau",
        "learning_rate_plateau_patience": train.LR_PLATEAU_PATIENCE,
        "minimum_learning_rate": train.MIN_LEARNING_RATE,
        "early_stopping_patience": train.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": train.EARLY_STOPPING_MIN_DELTA,
        "early_stopping_min_epochs": train.EARLY_STOPPING_MIN_EPOCHS,
        "max_samples": train.MAX_SAMPLES,
        "balance_classes": train.BALANCE_CLASSES,
        "use_feedback_data": False,
        "feedback_samples": 0,
        "test_ratio": train.TEST_RATIO,
        "validation_ratio": train.VAL_RATIO,
        "random_seed": train.RANDOM_SEED,
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "metadata_source": "evaluate.py",
        "backend": "NumPy CPU",
    })


def load_model(class_labels):
    if not Path(train.MODEL_PATH).exists():
        raise SystemExit(
            f"Model '{train.MODEL_PATH}' was not found. Run train.py before evaluate.py."
        )
    model = StandardCNN(
        input_dim=train.INPUT_DIM,
        num_classes=len(class_labels),
        conv_padding=train.CONV_PADDING,
        hidden_nodes=train.DENSE_HIDDEN_NODES,
    )
    model.load_weights(train.MODEL_PATH)
    if model.class_labels is None or not np.array_equal(model.class_labels, class_labels):
        raise ValueError("Saved model labels do not match the current dataset labels.")
    return model


def predict_dataset(model, X, y=None):
    predictions = np.empty(len(X), dtype=np.int32)
    confidence = np.empty(len(X), dtype=np.float64)
    losses = np.empty(len(X), dtype=np.float64) if y is not None else None

    start = time.perf_counter()
    for index, image in enumerate(X):
        probabilities = model.forward(image)
        predictions[index] = int(np.argmax(probabilities))
        confidence[index] = float(probabilities[predictions[index]])
        if y is not None:
            losses[index] = -np.log(probabilities[int(y[index])] + 1e-9)
    elapsed = time.perf_counter() - start

    return {
        "predictions": predictions,
        "confidence": confidence,
        "losses": losses,
        "elapsed_seconds": elapsed,
        "mean_inference_ms": (elapsed / max(1, len(X))) * 1000,
    }


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true.astype(int), y_pred.astype(int)), 1)
    return matrix


def classification_metrics(matrix):
    true_positive = np.diag(matrix).astype(np.float64)
    predicted_count = matrix.sum(axis=0).astype(np.float64)
    true_count = matrix.sum(axis=1).astype(np.float64)

    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros_like(true_positive),
        where=predicted_count != 0,
    )
    recall = np.divide(
        true_positive,
        true_count,
        out=np.zeros_like(true_positive),
        where=true_count != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    accuracy = true_positive.sum() / max(1, matrix.sum())

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_count.astype(np.int64),
    }


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def save_per_class_metrics(class_labels, matrix, metrics):
    rows = []
    for index, label in enumerate(class_labels):
        rows.append({
            "class_index": index,
            "label": chr(int(label)),
            "unicode": int(label),
            "precision": float(metrics["precision"][index]),
            "recall": float(metrics["recall"][index]),
            "f1": float(metrics["f1"][index]),
            "support": int(metrics["support"][index]),
            "correct": int(matrix[index, index]),
        })
    write_csv(
        OUTPUT_DIR / "per_class_metrics.csv",
        ["class_index", "label", "unicode", "precision", "recall", "f1", "support", "correct"],
        rows,
    )


def save_top_confusions(class_labels, matrix, limit=30):
    rows = []
    for true_index in range(len(class_labels)):
        for predicted_index in range(len(class_labels)):
            count = int(matrix[true_index, predicted_index])
            if true_index != predicted_index and count > 0:
                rows.append({
                    "true_label": chr(int(class_labels[true_index])),
                    "predicted_label": chr(int(class_labels[predicted_index])),
                    "count": count,
                    "rate_within_true_class": count / max(1, int(matrix[true_index].sum())),
                })
    rows.sort(key=lambda row: (-row["count"], -row["rate_within_true_class"]))
    write_csv(
        OUTPUT_DIR / "top_confusions.csv",
        ["true_label", "predicted_label", "count", "rate_within_true_class"],
        rows[:limit],
    )


def plot_training_curves():
    history_path = Path("results") / "baseline" / "training_history.csv"
    if not history_path.exists():
        print("Skipping learning curves: training_history.csv was not found.")
        return

    with history_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return
    if not TRAINING_HISTORY_COLUMNS.issubset(rows[0].keys()):
        print("Skipping learning curves: training_history.csv is from an older run.")
        stale_curve = OUTPUT_DIR / "training_curves.png"
        if stale_curve.exists():
            stale_curve.unlink()
        return

    epochs = [int(row["epoch"]) for row in rows]
    train_loss = [float(row["train_loss"]) for row in rows]
    val_loss = [float(row["validation_loss"]) for row in rows]
    train_acc = [float(row["train_accuracy"]) for row in rows]
    val_acc = [float(row["validation_accuracy"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(epochs, train_loss, marker="o", label="Train")
    axes[0].plot(epochs, val_loss, marker="o", label="Validation")
    axes[0].set(title="Loss by Epoch", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    axes[1].plot(epochs, train_acc, marker="o", label="Train")
    axes[1].plot(epochs, val_acc, marker="o", label="Validation")
    axes[1].set(title="Accuracy by Epoch", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=180)
    plt.close(fig)


def plot_class_distribution(class_labels):
    split_path = Path("results") / "baseline" / "split_summary.csv"
    if not split_path.exists():
        return
    with split_path.open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    labels = [row["label"] for row in rows]
    train_counts = [int(row["base_train_count"]) for row in rows]
    val_counts = [int(row["validation_count"]) for row in rows]
    test_counts = [int(row["test_count"]) for row in rows]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x, train_counts, label="Train", color="#35618f")
    ax.bar(x, val_counts, bottom=train_counts, label="Validation", color="#e2a23a")
    bottoms = np.asarray(train_counts) + np.asarray(val_counts)
    ax.bar(x, test_counts, bottom=bottoms, label="Test", color="#4f9d69")
    ax.set(title="Balanced Class Distribution", xlabel="Class", ylabel="Number of Images")
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "class_distribution.png", dpi=180)
    plt.close(fig)


def plot_confusion_matrices(class_labels, matrix):
    labels = [chr(int(label)) for label in class_labels]
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(
        matrix,
        row_sums,
        out=np.zeros_like(matrix, dtype=np.float64),
        where=row_sums != 0,
    )

    for values, title, filename, colorbar_label in [
        (matrix, "Confusion Matrix (Counts)", "confusion_matrix_counts.png", "Images"),
        (normalized, "Normalized Confusion Matrix", "confusion_matrix_normalized.png", "Rate"),
    ]:
        fig, ax = plt.subplots(figsize=(15, 13))
        image = ax.imshow(values, cmap="Blues", aspect="auto")
        ax.set(title=title, xlabel="Predicted Label", ylabel="True Label")
        ax.set_xticks(np.arange(len(labels)), labels, fontsize=7)
        ax.set_yticks(np.arange(len(labels)), labels, fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label=colorbar_label)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / filename, dpi=180)
        plt.close(fig)


def plot_f1_scores(class_labels, f1):
    labels = [chr(int(label)) for label in class_labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x, f1, color="#35618f")
    ax.axhline(float(np.mean(f1)), color="#b34a3c", linestyle="--", label="Macro-F1")
    ax.set(title="F1-Score by Class", xlabel="Class", ylabel="F1-Score", ylim=(0, 1.05))
    ax.set_xticks(x, labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "f1_by_class.png", dpi=180)
    plt.close(fig)


def plot_misclassified_examples(X, y_true, y_pred, confidence, class_labels):
    error_indices = np.where(y_true != y_pred)[0][:MAX_ERROR_IMAGES]
    if len(error_indices) == 0:
        return

    columns = 5
    rows = int(np.ceil(len(error_indices) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 2.7 * rows))
    axes = np.asarray(axes).reshape(-1)
    for axis in axes:
        axis.axis("off")
    for axis, sample_index in zip(axes, error_indices):
        true_label = chr(int(class_labels[int(y_true[sample_index])]))
        predicted_label = chr(int(class_labels[int(y_pred[sample_index])]))
        axis.imshow(X[sample_index], cmap="gray", vmin=0, vmax=255)
        axis.set_title(f"True {true_label} | Pred {predicted_label}\nConf {confidence[sample_index] * 100:.1f}%")
        axis.axis("off")
    fig.suptitle("Misclassified Test Images")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "misclassified_examples.png", dpi=180)
    plt.close(fig)


def shift_image(image, dx, dy):
    shifted = np.zeros_like(image)
    source_y_start = max(0, -dy)
    source_y_end = min(image.shape[0], image.shape[0] - dy)
    source_x_start = max(0, -dx)
    source_x_end = min(image.shape[1], image.shape[1] - dx)
    target_y_start = max(0, dy)
    target_y_end = target_y_start + (source_y_end - source_y_start)
    target_x_start = max(0, dx)
    target_x_end = target_x_start + (source_x_end - source_x_start)
    shifted[target_y_start:target_y_end, target_x_start:target_x_end] = image[
        source_y_start:source_y_end,
        source_x_start:source_x_end,
    ]
    return shifted


def center_image(image):
    coordinates = np.argwhere(image > 0)
    if len(coordinates) == 0:
        return image.copy()
    y_min, x_min = coordinates.min(axis=0)
    y_max, x_max = coordinates.max(axis=0)
    cropped = image[y_min:y_max + 1, x_min:x_max + 1]
    centered = np.zeros_like(image)
    height, width = cropped.shape
    start_y = (image.shape[0] - height) // 2
    start_x = (image.shape[1] - width) // 2
    centered[start_y:start_y + height, start_x:start_x + width] = cropped
    return centered


def evaluate_shift_robustness(model, X_test, y_test):
    rng = np.random.default_rng(train.RANDOM_SEED + 100)
    shifted = np.empty_like(X_test)
    for index, image in enumerate(X_test):
        dx, dy = rng.integers(-2, 3, size=2)
        shifted[index] = shift_image(image, int(dx), int(dy))
    centered = np.asarray([center_image(image) for image in shifted], dtype=X_test.dtype)

    conditions = []
    for name, images in [
        ("original", X_test),
        ("random_shift_-2_to_2", shifted),
        ("shifted_then_centered", centered),
    ]:
        result = predict_dataset(model, images, y_test)
        accuracy = float(np.mean(result["predictions"] == y_test))
        conditions.append({
            "condition": name,
            "accuracy": accuracy,
            "mean_loss": float(np.mean(result["losses"])),
        })

    write_csv(
        OUTPUT_DIR / "shift_robustness.csv",
        ["condition", "accuracy", "mean_loss"],
        conditions,
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [row["condition"].replace("_", " ") for row in conditions]
    values = [row["accuracy"] * 100 for row in conditions]
    bars = ax.bar(names, values, color=["#35618f", "#b34a3c", "#4f9d69"])
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    ax.set(title="Robustness to Spatial Shifts", ylabel="Accuracy (%)", ylim=(0, 105))
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "shift_robustness.png", dpi=180)
    plt.close(fig)


def plot_feature_maps(model, image, true_label):
    normalized = (image / 255.0) - 0.5
    conv1 = model.layers[0].forward(normalized)
    relu1 = model.layers[1].forward(conv1)
    pool1 = model.layers[2].forward(relu1)
    conv2 = model.layers[3].forward(pool1)
    relu2 = model.layers[4].forward(conv2)

    fig, axes = plt.subplots(3, 8, figsize=(14, 5.7))
    for axis in axes.reshape(-1):
        axis.axis("off")
    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title(f"Input: {true_label}")
    for index in range(8):
        axes[1, index].imshow(relu1[:, :, index], cmap="viridis")
        axes[1, index].set_title(f"Conv1 #{index + 1}")
        axes[2, index].imshow(relu2[:, :, index], cmap="viridis")
        axes[2, index].set_title(f"Conv2 #{index + 1}")
    fig.suptitle("Representative Feature Maps")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_maps.png", dpi=180)
    plt.close(fig)


def model_parameter_count(model):
    arrays = [
        model.layers[0].filters, model.layers[0].biases,
        model.layers[3].filters, model.layers[3].biases,
        model.layers[7].weights, model.layers[7].biases,
        model.layers[9].weights, model.layers[9].biases,
    ]
    return int(sum(array.size for array in arrays))


def write_results_readme():
    content = """CNN evaluation output

baseline/training_history.csv: train and validation metrics for every epoch.
baseline/run_config.json: complete baseline configuration.
baseline/split_summary.csv: class counts in train, validation, and test.
baseline/evaluation/metrics_summary.json: final test metrics.
baseline/evaluation/per_class_metrics.csv: precision, recall, and F1 for every class.
baseline/evaluation/top_confusions.csv: most frequent incorrect label pairs.
baseline/evaluation/test_predictions.npz: test images, labels, predictions, and confidence.
baseline/evaluation/shift_robustness.csv: accuracy before and after artificial shifts.
baseline/evaluation/*.png: report-ready charts and example images.
experiments/: optional sample-size, learning-rate, and padding comparisons.

No user-feedback comparison is included in these evaluation results.
"""
    readme_path = Path("results") / "README_RESULTS.txt"
    readme_path.write_text(content, encoding="utf-8")


def package_results():
    with zipfile.ZipFile(ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        results_root = Path("results")
        if results_root.exists():
            for path in results_root.rglob("*"):
                if path.is_file():
                    archive.write(path, path.as_posix())
        model_path = Path(train.MODEL_PATH)
        if model_path.exists():
            archive.write(model_path, model_path.name)
        source_files = [
            "layers.py",
            "cnn_models.py",
            "train.py",
            "evaluate.py",
            "run_experiments.py",
            "run_all_evaluations.py",
            "test_brain.py",
        ]
        for source_name in source_files:
            source_path = Path(source_name)
            if source_path.exists():
                archive.write(source_path, f"source/{source_name}")
    print("Created download archive:", ARCHIVE_PATH)


def main():
    require_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    remove_stale_training_artifacts()
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels = prepare_data_splits()
    sync_baseline_metadata(X_train, X_val, X_test, y_train, y_val, y_test, class_labels)
    model = load_model(class_labels)

    for image in X_test[:INFERENCE_WARMUP_IMAGES]:
        model.forward(image)

    prediction_result = predict_dataset(model, X_test, y_test)
    y_pred = prediction_result["predictions"]
    matrix = confusion_matrix(y_test, y_pred, len(class_labels))
    metrics = classification_metrics(matrix)

    np.savez_compressed(
        OUTPUT_DIR / "test_predictions.npz",
        X=X_test,
        y_true=y_test,
        y_pred=y_pred,
        confidence=prediction_result["confidence"],
        class_labels=class_labels,
    )
    np.savetxt(OUTPUT_DIR / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    summary = {
        "test_samples": len(X_test),
        "test_loss": float(np.mean(prediction_result["losses"])),
        "test_accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "mean_confidence": float(np.mean(prediction_result["confidence"])),
        "mean_inference_ms": prediction_result["mean_inference_ms"],
        "model_parameters": model_parameter_count(model),
        "model_file_bytes": Path(train.MODEL_PATH).stat().st_size,
    }
    write_json(OUTPUT_DIR / "metrics_summary.json", summary)
    save_per_class_metrics(class_labels, matrix, metrics)
    save_top_confusions(class_labels, matrix)

    plot_training_curves()
    plot_class_distribution(class_labels)
    plot_confusion_matrices(class_labels, matrix)
    plot_f1_scores(class_labels, metrics["f1"])
    plot_misclassified_examples(
        X_test,
        y_test,
        y_pred,
        prediction_result["confidence"],
        class_labels,
    )
    evaluate_shift_robustness(model, X_test, y_test)
    plot_feature_maps(model, X_test[0], chr(int(class_labels[int(y_test[0])])))

    write_results_readme()
    package_results()
    print("Evaluation results saved to", OUTPUT_DIR)
    print("Test accuracy:", round(metrics["accuracy"] * 100, 2), "%")
    print("Macro-F1:", round(metrics["macro_f1"], 4))


if __name__ == "__main__":
    main()
