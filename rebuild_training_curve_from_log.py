import csv
import json
import os
import tempfile
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROWS = [
    (1, 0.008, 0.008, False, 0.7639, 74.92, 0.8570, 71.83, 2662.61, True),
    (2, 0.008, 0.008, False, 0.6105, 79.64, 0.7757, 76.00, 2847.52, True),
    (3, 0.008, 0.008, False, 0.5189, 81.86, 0.6820, 77.11, 2733.05, True),
    (4, 0.008, 0.008, False, 0.4624, 83.96, 0.6476, 79.36, 2771.23, True),
    (5, 0.008, 0.004, True, 0.4273, 84.50, 0.6511, 78.21, 2719.67, False),
    (6, 0.004, 0.004, False, 0.2922, 89.34, 0.5653, 81.15, 2750.49, True),
    (7, 0.004, 0.002, True, 0.2728, 89.85, 0.6043, 81.06, 2709.08, False),
]

FIELDS = [
    "epoch",
    "learning_rate",
    "next_learning_rate",
    "learning_rate_reduced",
    "train_loss",
    "train_accuracy",
    "validation_loss",
    "validation_accuracy",
    "epoch_seconds",
    "is_best_checkpoint",
]


def as_dicts():
    return [dict(zip(FIELDS, row)) for row in ROWS]


def font(size=20, bold=False):
    names = ["arialbd.ttf" if bold else "arial.ttf", "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def plot_panel(draw, box, title, ylabel, series, y_min=None, y_max=None, best_epoch=6):
    left, top, right, bottom = box
    title_font = font(24, True)
    label_font = font(16)
    small_font = font(14)

    draw.text((left, top - 44), title, fill="#1f2933", font=title_font)
    plot_left = left + 58
    plot_top = top + 10
    plot_right = right - 20
    plot_bottom = bottom - 44

    values = [value for _, points, _ in series for value in points]
    y_min = min(values) if y_min is None else y_min
    y_max = max(values) if y_max is None else y_max
    padding = (y_max - y_min) * 0.08 or 1
    y_min -= padding
    y_max += padding

    def x_pos(epoch):
        return plot_left + (epoch - 1) * (plot_right - plot_left) / 6

    def y_pos(value):
        return plot_bottom - (value - y_min) * (plot_bottom - plot_top) / (y_max - y_min)

    for i in range(5):
        y = plot_top + i * (plot_bottom - plot_top) / 4
        draw.line((plot_left, y, plot_right, y), fill="#e6ebf0", width=1)
        tick = y_max - i * (y_max - y_min) / 4
        draw.text((left, y - 9), f"{tick:.2f}", fill="#52616b", font=small_font)

    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#52616b", width=2)
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#52616b", width=2)
    draw.text((plot_left + 150, bottom - 24), "Epoch", fill="#52616b", font=label_font)
    draw.text((left, top - 18), ylabel, fill="#52616b", font=label_font)

    best_x = x_pos(best_epoch)
    draw.line((best_x, plot_top, best_x, plot_bottom), fill="#4f9d69", width=2)
    draw.text((best_x + 5, plot_top + 6), "Best epoch 6", fill="#2f7d4f", font=small_font)

    for epoch in range(1, 8):
        x = x_pos(epoch)
        draw.line((x, plot_bottom, x, plot_bottom + 6), fill="#52616b", width=1)
        draw.text((x - 5, plot_bottom + 10), str(epoch), fill="#52616b", font=small_font)

    for name, points, color in series:
        coords = [(x_pos(i + 1), y_pos(value)) for i, value in enumerate(points)]
        for start, end in zip(coords, coords[1:]):
            draw.line((*start, *end), fill=color, width=4)
        for x, y in coords:
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)

    legend_x = plot_right - 180
    legend_y = top - 36
    for offset, (name, _, color) in enumerate(series):
        y = legend_y + offset * 22
        draw.line((legend_x, y + 8, legend_x + 28, y + 8), fill=color, width=4)
        draw.text((legend_x + 36, y), name, fill="#1f2933", font=small_font)


def draw_training_curve(path):
    rows = as_dicts()
    image = Image.new("RGB", (2160, 810), "white")
    draw = ImageDraw.Draw(image)
    draw.text((60, 36), "Training Curves", fill="#1f2933", font=font(34, True))
    draw.text(
        (60, 78),
        "Dense-64 CNN, 23,500 selected images. Vertical marker shows the best validation-loss checkpoint.",
        fill="#52616b",
        font=font(18),
    )

    train_loss = [row["train_loss"] for row in rows]
    val_loss = [row["validation_loss"] for row in rows]
    train_acc = [row["train_accuracy"] for row in rows]
    val_acc = [row["validation_accuracy"] for row in rows]

    plot_panel(
        draw,
        (70, 180, 1050, 740),
        "Loss by Epoch",
        "Cross-Entropy Loss",
        [("Train", train_loss, "#35618f"), ("Validation", val_loss, "#b34a3c")],
    )
    plot_panel(
        draw,
        (1120, 180, 2100, 740),
        "Accuracy by Epoch",
        "Accuracy (%)",
        [("Train", train_acc, "#35618f"), ("Validation", val_acc, "#b34a3c")],
        y_min=68,
        y_max=92,
    )
    image.save(path)


def update_zip(zip_path, replacements):
    fd, temp_name = tempfile.mkstemp(suffix=".zip", dir=".")
    os.close(fd)
    replacement_names = set(replacements)

    with zipfile.ZipFile(zip_path, "r") as old_zip:
        with zipfile.ZipFile(temp_name, "w", compression=zipfile.ZIP_DEFLATED) as new_zip:
            for info in old_zip.infolist():
                if info.filename not in replacement_names:
                    new_zip.writestr(info, old_zip.read(info.filename))
            for archive_name, source_path in replacements.items():
                new_zip.write(source_path, archive_name)

    os.replace(temp_name, zip_path)


def main():
    rows = as_dicts()
    base_dir = Path("results") / "baseline"
    eval_dir = base_dir / "evaluation"
    base_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    history_path = base_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "test_loss": 0.6088,
        "test_accuracy": 80.89,
        "total_training_seconds": sum(row["epoch_seconds"] for row in rows),
        "completed_epochs": 7,
        "best_epoch": 6,
        "best_validation_loss": 0.5653,
        "best_validation_accuracy": 81.15,
        "stopped_early": False,
        "final_learning_rate": 0.002,
        "learning_rate_reductions": 2,
        "dense_hidden_nodes": 64,
        "model_parameters": 40687,
        "model_path": "emnist_47_brain.npz",
        "metadata_source": "training_log",
    }
    summary_path = base_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    curve_path = eval_dir / "training_curves.png"
    draw_training_curve(curve_path)

    update_zip(
        Path("cnn_evaluation_results.zip"),
        {
            "results/baseline/training_history.csv": history_path,
            "results/baseline/training_summary.json": summary_path,
            "results/baseline/evaluation/training_curves.png": curve_path,
        },
    )
    print("Updated cnn_evaluation_results.zip with training curve and training records.")


if __name__ == "__main__":
    main()
