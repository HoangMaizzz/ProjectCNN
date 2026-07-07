import csv
import json
import time
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as error:
    matplotlib = None
    plt = None

import evaluate
import train
from cnn_models import StandardCNN


OUTPUT_DIR = Path("results") / "experiments"

# These switches let Kaggle skip an expensive experiment when needed.
RUN_SAMPLE_SIZE_EXPERIMENT = True
RUN_LEARNING_RATE_EXPERIMENT = True
RUN_PADDING_EXPERIMENT = True

COMPARISON_EPOCHS = 10
SAMPLE_SIZES = [2350, 4700, 7050, 10000]
LEARNING_RATES = [0.002, 0.004, 0.008, 0.012]
PADDINGS = [0, 1]
CONTROL_MAX_SAMPLES = 4700


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def prepare_data(max_samples):
    dataset = np.load(train.resolve_data_path(train.DATA_PATH), allow_pickle=False)
    X, y = train.select_subset(
        dataset["X"],
        dataset["y"],
        max_samples=max_samples,
        balance_classes=True,
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


def capture_model_state(model):
    return {
        "conv1_filters": model.layers[0].filters.copy(),
        "conv1_biases": model.layers[0].biases.copy(),
        "conv2_filters": model.layers[3].filters.copy(),
        "conv2_biases": model.layers[3].biases.copy(),
        "dense1_weights": model.layers[7].weights.copy(),
        "dense1_biases": model.layers[7].biases.copy(),
        "dense2_weights": model.layers[9].weights.copy(),
        "dense2_biases": model.layers[9].biases.copy(),
    }


def restore_model_state(model, state):
    model.layers[0].filters = state["conv1_filters"]
    model.layers[0].biases = state["conv1_biases"]
    model.layers[3].filters = state["conv2_filters"]
    model.layers[3].biases = state["conv2_biases"]
    model.layers[7].weights = state["dense1_weights"]
    model.layers[7].biases = state["dense1_biases"]
    model.layers[9].weights = state["dense2_weights"]
    model.layers[9].biases = state["dense2_biases"]


def train_configuration(max_samples, learning_rate, padding, epochs):
    np.random.seed(train.RANDOM_SEED)
    X_train, X_val, X_test, y_train, y_val, y_test, class_labels = prepare_data(max_samples)
    model = StandardCNN(
        input_dim=train.INPUT_DIM,
        num_classes=len(class_labels),
        conv_padding=padding,
        hidden_nodes=train.DENSE_HIDDEN_NODES,
    )

    history = []
    best_state = None
    best_validation_loss = float("inf")
    best_validation_accuracy = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    plateau_epochs = 0
    current_learning_rate = learning_rate
    learning_rate_reductions = 0
    stopped_early = False
    total_start = time.perf_counter()
    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        permutation = np.random.permutation(len(X_train))
        online_loss = 0.0
        online_correct = 0

        for position, sample_index in enumerate(permutation, start=1):
            loss, correct = model.train_step(
                X_train[sample_index],
                int(y_train[sample_index]),
                learning_rate=current_learning_rate,
            )
            online_loss += float(loss)
            online_correct += int(correct)
            if position % 500 == 0:
                print("  training image", position, "/", len(X_train))

        validation_loss, validation_accuracy = train.evaluate_model(model, X_val, y_val)
        epoch_seconds = time.perf_counter() - epoch_start
        is_best = validation_loss < (
            best_validation_loss - train.EARLY_STOPPING_MIN_DELTA
        )
        if is_best:
            best_validation_loss = float(validation_loss)
            best_validation_accuracy = float(validation_accuracy)
            best_epoch = epoch + 1
            best_state = capture_model_state(model)
            epochs_without_improvement = 0
            plateau_epochs = 0
        else:
            epochs_without_improvement += 1
            plateau_epochs += 1

        next_learning_rate = current_learning_rate
        learning_rate_reduced = False
        if (
            not is_best
            and plateau_epochs >= train.LR_PLATEAU_PATIENCE
            and current_learning_rate > train.MIN_LEARNING_RATE
        ):
            next_learning_rate = train.decay_learning_rate(current_learning_rate)
            learning_rate_reduced = next_learning_rate < current_learning_rate
            plateau_epochs = 0
            if learning_rate_reduced:
                learning_rate_reductions += 1

        history.append({
            "epoch": epoch + 1,
            "learning_rate": float(current_learning_rate),
            "next_learning_rate": float(next_learning_rate),
            "learning_rate_reduced": bool(learning_rate_reduced),
            "online_train_loss": online_loss / len(X_train),
            "online_train_accuracy": (online_correct / len(X_train)) * 100,
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(validation_accuracy),
            "epoch_seconds": epoch_seconds,
            "is_best_checkpoint": bool(is_best),
        })
        print(
            "Epoch", epoch + 1, "/", epochs,
            "lr=", current_learning_rate,
            "val_acc=", round(validation_accuracy, 2),
            "time=", round(epoch_seconds, 2), "s",
        )

        current_learning_rate = next_learning_rate

        if (
            epoch + 1 >= train.EARLY_STOPPING_MIN_EPOCHS
            and epochs_without_improvement >= train.EARLY_STOPPING_PATIENCE
        ):
            stopped_early = True
            print("Early stopping at epoch", epoch + 1, "best epoch:", best_epoch)
            break

    if best_state is None:
        raise RuntimeError("Experiment did not produce a finite validation checkpoint.")
    restore_model_state(model, best_state)

    prediction_result = evaluate.predict_dataset(model, X_test, y_test)
    matrix = evaluate.confusion_matrix(
        y_test,
        prediction_result["predictions"],
        len(class_labels),
    )
    metrics = evaluate.classification_metrics(matrix)
    total_seconds = time.perf_counter() - total_start

    summary = {
        "max_samples_requested": max_samples,
        "samples_used": len(X_train) + len(X_val) + len(X_test),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "learning_rate": learning_rate,
        "learning_rate_decay_factor": train.LR_DECAY_FACTOR,
        "learning_rate_scheduler": "reduce_on_validation_loss_plateau",
        "learning_rate_plateau_patience": train.LR_PLATEAU_PATIENCE,
        "minimum_learning_rate": train.MIN_LEARNING_RATE,
        "final_learning_rate": float(current_learning_rate),
        "learning_rate_reductions": learning_rate_reductions,
        "padding": padding,
        "hidden_nodes": train.DENSE_HIDDEN_NODES,
        "epochs": epochs,
        "completed_epochs": len(history),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "best_validation_accuracy": best_validation_accuracy,
        "stopped_early": stopped_early,
        "early_stopping_min_epochs": train.EARLY_STOPPING_MIN_EPOCHS,
        "final_validation_loss": history[-1]["validation_loss"],
        "final_validation_accuracy": history[-1]["validation_accuracy"],
        "test_loss": float(np.mean(prediction_result["losses"])),
        "test_accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "training_and_test_seconds": total_seconds,
        "model_parameters": evaluate.model_parameter_count(model),
        "mean_inference_ms": prediction_result["mean_inference_ms"],
    }
    return summary, history


def experiment_key(max_samples, learning_rate, padding, epochs):
    return (int(max_samples), float(learning_rate), int(padding), int(epochs))


def run_or_reuse(cache, max_samples, learning_rate, padding, epochs):
    key = experiment_key(max_samples, learning_rate, padding, epochs)
    if key in cache:
        print("Reusing completed configuration:", key)
        return cache[key]

    run_id = (
        f"samples_{max_samples}_lr_{learning_rate:g}_pad_{padding}_epochs_{epochs}"
        f"_h{train.DENSE_HIDDEN_NODES}_decay{train.LR_DECAY_FACTOR:g}"
        f"_plateau{train.LR_PLATEAU_PATIENCE}_minlr{train.MIN_LEARNING_RATE:g}"
        f"_es{train.EARLY_STOPPING_PATIENCE}"
        f"_min{train.EARLY_STOPPING_MIN_EPOCHS}"
    )
    history_path = OUTPUT_DIR / "histories" / f"{run_id}.csv"
    summary_path = OUTPUT_DIR / "summaries" / f"{run_id}.json"
    if history_path.exists() and summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        with history_path.open("r", encoding="utf-8") as file:
            history = list(csv.DictReader(file))
        numeric_history_fields = [
            "learning_rate", "next_learning_rate", "online_train_loss",
            "online_train_accuracy", "validation_loss",
            "validation_accuracy", "epoch_seconds",
        ]
        for row in history:
            row["epoch"] = int(row["epoch"])
            for field in numeric_history_fields:
                row[field] = float(row[field])
        cache[key] = (summary, history)
        print("Loaded completed configuration:", run_id)
        return summary, history

    print("\nRunning", run_id)
    summary, history = train_configuration(max_samples, learning_rate, padding, epochs)
    summary["run_id"] = run_id
    for row in history:
        row["run_id"] = run_id
    write_csv(
        history_path,
        [
            "run_id", "epoch", "learning_rate", "next_learning_rate",
            "learning_rate_reduced", "online_train_loss", "online_train_accuracy",
            "validation_loss", "validation_accuracy", "epoch_seconds",
            "is_best_checkpoint",
        ],
        history,
    )
    evaluate.write_json(summary_path, summary)
    cache[key] = (summary, history)
    return summary, history


def plot_sample_size(rows):
    rows = sorted(rows, key=lambda row: row["samples_used"])
    samples = [row["samples_used"] for row in rows]
    accuracy = [row["test_accuracy"] * 100 for row in rows]
    macro_f1 = [row["macro_f1"] * 100 for row in rows]
    seconds = [row["training_and_test_seconds"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(samples, accuracy, marker="o", label="Test Accuracy")
    axes[0].plot(samples, macro_f1, marker="o", label="Macro-F1")
    axes[0].set(title="Performance vs. Dataset Size", xlabel="Images", ylabel="Score (%)")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].bar([str(value) for value in samples], seconds, color="#35618f")
    axes[1].set(title="Runtime vs. Dataset Size", xlabel="Images", ylabel="Seconds")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_size_comparison.png", dpi=180)
    plt.close(fig)


def plot_learning_rates(rows, histories):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in sorted(rows, key=lambda item: item["learning_rate"]):
        history = histories[row["run_id"]]
        axes[0].plot(
            [item["epoch"] for item in history],
            [item["validation_loss"] for item in history],
            marker="o",
            label=f"LR={row['learning_rate']:g}",
        )
    axes[0].set(title="Validation Loss by Learning Rate", xlabel="Epoch", ylabel="Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    labels = [f"{row['learning_rate']:g}" for row in rows]
    values = [row["test_accuracy"] * 100 for row in rows]
    bars = axes[1].bar(labels, values, color="#4f9d69")
    axes[1].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[1].set(title="Test Accuracy by Learning Rate", xlabel="Learning Rate", ylabel="Accuracy (%)")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "learning_rate_comparison.png", dpi=180)
    plt.close(fig)


def plot_padding(rows):
    rows = sorted(rows, key=lambda row: row["padding"])
    labels = [f"Padding {row['padding']}" for row in rows]
    x = np.arange(len(rows))
    width = 0.34

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(x - width / 2, [row["test_accuracy"] * 100 for row in rows], width, label="Accuracy")
    axes[0].bar(x + width / 2, [row["macro_f1"] * 100 for row in rows], width, label="Macro-F1")
    axes[0].set(title="Performance by Padding", ylabel="Score (%)", xticks=x, xticklabels=labels)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.2)

    axes[1].bar(labels, [row["model_parameters"] for row in rows], color="#e2a23a")
    axes[1].set(title="Parameter Count by Padding", ylabel="Parameters")
    axes[1].grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "padding_comparison.png", dpi=180)
    plt.close(fig)


def main():
    if plt is None:
        raise SystemExit(
            "Experiment charts require matplotlib. Install it with: pip install matplotlib"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluate.write_json(OUTPUT_DIR / "experiment_config.json", {
        "comparison_epochs": COMPARISON_EPOCHS,
        "sample_sizes": SAMPLE_SIZES,
        "learning_rates": LEARNING_RATES,
        "paddings": PADDINGS,
        "control_max_samples": CONTROL_MAX_SAMPLES,
        "dense_hidden_nodes": train.DENSE_HIDDEN_NODES,
        "learning_rate_decay_factor": train.LR_DECAY_FACTOR,
        "learning_rate_scheduler": "reduce_on_validation_loss_plateau",
        "learning_rate_plateau_patience": train.LR_PLATEAU_PATIENCE,
        "minimum_learning_rate": train.MIN_LEARNING_RATE,
        "early_stopping_patience": train.EARLY_STOPPING_PATIENCE,
        "early_stopping_min_delta": train.EARLY_STOPPING_MIN_DELTA,
        "early_stopping_min_epochs": train.EARLY_STOPPING_MIN_EPOCHS,
        "random_seed": train.RANDOM_SEED,
        "run_sample_size_experiment": RUN_SAMPLE_SIZE_EXPERIMENT,
        "run_learning_rate_experiment": RUN_LEARNING_RATE_EXPERIMENT,
        "run_padding_experiment": RUN_PADDING_EXPERIMENT,
        "feedback_used": False,
    })
    cache = {}
    all_rows = []
    histories = {}

    if RUN_SAMPLE_SIZE_EXPERIMENT:
        sample_rows = []
        for max_samples in SAMPLE_SIZES:
            summary, history = run_or_reuse(
                cache,
                max_samples,
                train.LEARNING_RATE,
                train.CONV_PADDING,
                COMPARISON_EPOCHS,
            )
            sample_rows.append(summary)
            histories[summary["run_id"]] = history
        write_csv(OUTPUT_DIR / "sample_size_results.csv", list(sample_rows[0].keys()), sample_rows)
        plot_sample_size(sample_rows)
        all_rows.extend(sample_rows)

    if RUN_LEARNING_RATE_EXPERIMENT:
        learning_rate_rows = []
        for learning_rate in LEARNING_RATES:
            summary, history = run_or_reuse(
                cache,
                CONTROL_MAX_SAMPLES,
                learning_rate,
                train.CONV_PADDING,
                COMPARISON_EPOCHS,
            )
            learning_rate_rows.append(summary)
            histories[summary["run_id"]] = history
        write_csv(
            OUTPUT_DIR / "learning_rate_results.csv",
            list(learning_rate_rows[0].keys()),
            learning_rate_rows,
        )
        plot_learning_rates(learning_rate_rows, histories)
        all_rows.extend(learning_rate_rows)

    if RUN_PADDING_EXPERIMENT:
        padding_rows = []
        for padding in PADDINGS:
            summary, history = run_or_reuse(
                cache,
                CONTROL_MAX_SAMPLES,
                train.LEARNING_RATE,
                padding,
                COMPARISON_EPOCHS,
            )
            padding_rows.append(summary)
            histories[summary["run_id"]] = history
        write_csv(OUTPUT_DIR / "padding_results.csv", list(padding_rows[0].keys()), padding_rows)
        plot_padding(padding_rows)
        all_rows.extend(padding_rows)

    unique_rows = {row["run_id"]: row for row in all_rows}
    if unique_rows:
        final_rows = list(unique_rows.values())
        write_csv(OUTPUT_DIR / "all_experiments.csv", list(final_rows[0].keys()), final_rows)

    evaluate.write_results_readme()
    evaluate.package_results()
    print("Experiment results saved to", OUTPUT_DIR)
    print("Final download archive:", evaluate.ARCHIVE_PATH)


if __name__ == "__main__":
    main()
