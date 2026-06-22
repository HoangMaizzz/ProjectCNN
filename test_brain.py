import os
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import messagebox

import numpy as np

from cnn_models import StandardCNN


MODEL_PATH = "emnist_47_brain.npz"
MODEL_ARCHIVE_PATH = "cnn_evaluation_results.zip"
FEEDBACK_DATA_PATH = "user_feedback_47_classes_16x16.npz"


def prepare_model_file():
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        return model_path

    archive_path = Path(MODEL_ARCHIVE_PATH)
    if not archive_path.exists():
        raise SystemExit(
            f"Neither '{MODEL_PATH}' nor '{MODEL_ARCHIVE_PATH}' was found. "
            "Download the Kaggle evaluation ZIP or copy the trained model here."
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name == MODEL_PATH
        ]
        if not candidates:
            raise SystemExit(
                f"'{MODEL_ARCHIVE_PATH}' does not contain '{MODEL_PATH}'."
            )
        model_path.write_bytes(archive.read(candidates[0]))

    print("Extracted model from", MODEL_ARCHIVE_PATH)
    return model_path


model_path = prepare_model_file()

print("Loading labels from model...")
with np.load(model_path, allow_pickle=False) as saved_model:
    if "class_labels" not in saved_model.files:
        raise SystemExit("The trained model does not contain class labels.")
    unique_labels = np.asarray(saved_model["class_labels"])
    hidden_nodes = int(
        saved_model["hidden_nodes"].item()
        if "hidden_nodes" in saved_model.files
        else len(saved_model["dense1_biases"])
    )

if unique_labels.ndim != 1 or len(unique_labels) != 47:
    raise SystemExit("The trained model must contain exactly 47 class labels.")

valid_chars = {chr(int(label)): int(label) for label in unique_labels}

print("Loading model...")
model = StandardCNN(
    input_dim=16,
    num_classes=len(unique_labels),
    conv_padding=1,
    hidden_nodes=hidden_nodes,
)

try:
    model.load_weights(model_path)
except FileNotFoundError:
    raise SystemExit(
        f"Model file '{MODEL_PATH}' was not found. Train first with: python train.py"
    )

if model.class_labels is None or not np.array_equal(model.class_labels, unique_labels):
    raise SystemExit(
        "The saved model does not match the 47-class dataset. Train it again with: python train.py"
    )

print("Ready!")


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMNIST 16x16 Recognizer")

        self.cell_size = 24
        self.grid_size = 16
        self.image_data = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.last_prediction_image = None

        self.canvas = tk.Canvas(
            root,
            width=self.grid_size * self.cell_size,
            height=self.grid_size * self.cell_size,
            bg="black",
        )
        self.canvas.pack(pady=10)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<Button-3>", self.erase)

        button_frame = tk.Frame(root)
        button_frame.pack(pady=5)

        self.btn_predict = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            font=("Arial", 13, "bold"),
            fg="blue",
        )
        self.btn_predict.grid(row=0, column=0, padx=4)

        self.btn_clear = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            font=("Arial", 13),
        )
        self.btn_clear.grid(row=0, column=1, padx=4)

        self.lbl_result = tk.Label(
            root,
            text="Draw a digit, uppercase letter, or supported lowercase letter.",
            font=("Arial", 15),
        )
        self.lbl_result.pack(pady=8)

        feedback_frame = tk.Frame(root)
        feedback_frame.pack(pady=5)

        self.btn_correct = tk.Button(
            feedback_frame,
            text="Correct",
            command=self.confirm_correct,
            state=tk.DISABLED,
        )
        self.btn_correct.grid(row=0, column=0, padx=4)

        self.btn_wrong = tk.Button(
            feedback_frame,
            text="Wrong",
            command=self.enable_correction,
            state=tk.DISABLED,
        )
        self.btn_wrong.grid(row=0, column=1, padx=4)

        tk.Label(feedback_frame, text="Correct label:").grid(row=1, column=0, padx=4, pady=6)

        self.entry_correct = tk.Entry(feedback_frame, width=8, justify="center")
        self.entry_correct.grid(row=1, column=1, padx=4, pady=6)
        self.entry_correct.config(state=tk.DISABLED)

        self.btn_save = tk.Button(
            feedback_frame,
            text="Save fix",
            command=self.save_correction,
            state=tk.DISABLED,
        )
        self.btn_save.grid(row=1, column=2, padx=4, pady=6)

        self.lbl_feedback = tk.Label(root, text="", font=("Arial", 11))
        self.lbl_feedback.pack(pady=4)

    def draw(self, event):
        self.paint(event.x, event.y, 255, "white")

    def erase(self, event):
        self.paint(event.x, event.y, 0, "black")

    def paint(self, x, y, value, color):
        col = x // self.cell_size
        row = y // self.cell_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.image_data[row, col] = value
            x1 = col * self.cell_size
            y1 = row * self.cell_size
            x2 = x1 + self.cell_size
            y2 = y1 + self.cell_size
            self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="gray")

    def clear(self):
        self.image_data.fill(0)
        self.canvas.delete("all")
        self.last_prediction_image = None
        self.lbl_result.config(
            text="Draw a digit, uppercase letter, or supported lowercase letter.",
            fg="black",
        )
        self.lbl_feedback.config(text="")
        self.set_feedback_state(False)
        self.entry_correct.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.entry_correct.delete(0, tk.END)

    def center_image(self, image):
        coords = np.argwhere(image > 0)
        if len(coords) == 0:
            return image.copy()

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        cropped = image[y_min:y_max + 1, x_min:x_max + 1]

        centered = np.zeros((16, 16), dtype=np.uint8)
        h, w = cropped.shape
        start_y = (16 - h) // 2
        start_x = (16 - w) // 2
        centered[start_y:start_y + h, start_x:start_x + w] = cropped
        return centered

    def predict(self):
        if not np.any(self.image_data):
            messagebox.showinfo("Empty drawing", "Please draw something first.")
            return

        centered_data = self.center_image(self.image_data)
        pred_label, confidence = model.predict(centered_data)
        pred_char = chr(int(pred_label))

        self.last_prediction_image = centered_data
        self.lbl_result.config(
            text=f"Prediction: {pred_char}\nConfidence: {confidence:.2f}%",
            fg="red",
        )
        self.lbl_feedback.config(text="Was this prediction correct?")
        self.set_feedback_state(True)
        self.entry_correct.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)
        self.entry_correct.delete(0, tk.END)

    def set_feedback_state(self, enabled):
        state = tk.NORMAL if enabled else tk.DISABLED
        self.btn_correct.config(state=state)
        self.btn_wrong.config(state=state)

    def confirm_correct(self):
        self.lbl_feedback.config(text="Thanks. No correction saved.")
        self.set_feedback_state(False)

    def enable_correction(self):
        self.entry_correct.config(state=tk.NORMAL)
        self.btn_save.config(state=tk.NORMAL)
        self.entry_correct.focus_set()
        self.lbl_feedback.config(text="Enter the exact label, for example 7, A, or a.")

    def save_correction(self):
        if self.last_prediction_image is None:
            return

        label_text = self.entry_correct.get().strip()
        if len(label_text) != 1 or label_text not in valid_chars:
            allowed = "".join(valid_chars.keys())
            messagebox.showerror("Invalid label", f"Use one of these labels: {allowed}")
            return

        label_value = valid_chars[label_text]
        self.append_feedback_sample(self.last_prediction_image, label_value)

        self.lbl_feedback.config(
            text=f"Saved correction as '{label_text}'. It will be used next time you train."
        )
        self.set_feedback_state(False)
        self.entry_correct.config(state=tk.DISABLED)
        self.btn_save.config(state=tk.DISABLED)

    def append_feedback_sample(self, image, label):
        new_X = image.reshape(1, 16, 16).astype(np.uint8)
        new_y = np.array([label], dtype=np.int32)

        if os.path.exists(FEEDBACK_DATA_PATH):
            feedback = np.load(FEEDBACK_DATA_PATH)
            X_feedback = np.concatenate([feedback["X"], new_X], axis=0)
            y_feedback = np.concatenate([feedback["y"], new_y], axis=0)
        else:
            X_feedback = new_X
            y_feedback = new_y

        np.savez_compressed(FEEDBACK_DATA_PATH, X=X_feedback, y=y_feedback)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
