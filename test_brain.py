import os
import tkinter as tk
import zipfile
from pathlib import Path
from tkinter import messagebox

import numpy as np

from cnn_models import StandardCNN


MODEL_PATH = Path("models") / "emnist_47_brain.npz"
MODEL_ARCHIVE_PATH = Path("artifacts") / "cnn_evaluation_results.zip"
FEEDBACK_DATA_PATH = Path("data") / "user_feedback_47_classes_16x16.npz"
EXTRA_FEEDBACK_CHARS = ["ư"]


def prepare_model_file():
    model_path = Path(MODEL_PATH)
    for candidate in [model_path, Path(model_path.name)]:
        if candidate.exists():
            return candidate

    archive_candidates = [Path(MODEL_ARCHIVE_PATH), Path(Path(MODEL_ARCHIVE_PATH).name)]
    archive_path = next((path for path in archive_candidates if path.exists()), None)
    if archive_path is None:
        raise SystemExit(
            f"Neither '{MODEL_PATH}' nor '{MODEL_ARCHIVE_PATH}' was found. "
            "Download the Kaggle evaluation ZIP or copy the trained model here."
        )

    with zipfile.ZipFile(archive_path, "r") as archive:
        candidates = [
            name for name in archive.namelist()
            if Path(name).name == model_path.name
        ]
        if not candidates:
            raise SystemExit(
                f"'{MODEL_ARCHIVE_PATH}' does not contain '{MODEL_PATH}'."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(archive.read(candidates[0]))

    print("Extracted model from", archive_path)
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

if unique_labels.ndim != 1:
    raise SystemExit("The trained model must contain a one-dimensional class label list.")

valid_chars = {chr(int(label)): int(label) for label in unique_labels}
for char in EXTRA_FEEDBACK_CHARS:
    valid_chars.setdefault(char, ord(char))

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
        "The saved model labels do not match its metadata. Train it again with: python train.py"
    )

print("Ready!")


class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("EMNIST 16x16 Recognizer")
        self.root.configure(bg="#f3f4f6")
        self.root.minsize(900, 720)
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.attributes("-zoomed", True)

        self.grid_size = 16
        self.canvas_size = self.get_demo_canvas_size()
        self.cell_size = self.canvas_size // self.grid_size
        self.image_data = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        self.last_prediction_image = None

        self.main_frame = tk.Frame(root, bg="#f3f4f6")
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=24, pady=18)

        self.canvas = tk.Canvas(
            self.main_frame,
            width=self.canvas_size,
            height=self.canvas_size,
            bg="black",
            highlightthickness=2,
            highlightbackground="#111827",
        )
        self.canvas.pack(pady=(0, 18))

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<B3-Motion>", self.erase)
        self.canvas.bind("<Button-3>", self.erase)

        button_frame = tk.Frame(self.main_frame, bg="#f3f4f6")
        button_frame.pack(pady=(0, 12))

        self.btn_predict = tk.Button(
            button_frame,
            text="Predict",
            command=self.predict,
            font=("Arial", 16, "bold"),
            fg="blue",
            padx=18,
            pady=6,
        )
        self.btn_predict.grid(row=0, column=0, padx=8)

        self.btn_clear = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear,
            font=("Arial", 16),
            padx=18,
            pady=6,
        )
        self.btn_clear.grid(row=0, column=1, padx=8)

        self.lbl_result = tk.Label(
            self.main_frame,
            text="Draw a digit, uppercase letter, or supported lowercase letter.",
            font=("Arial", 20),
            bg="#f3f4f6",
        )
        self.lbl_result.pack(pady=(0, 14))

        feedback_frame = tk.Frame(self.main_frame, bg="#f3f4f6")
        feedback_frame.pack(pady=(0, 10))

        self.btn_correct = tk.Button(
            feedback_frame,
            text="Correct",
            command=self.confirm_correct,
            state=tk.DISABLED,
            font=("Arial", 13),
            padx=12,
            pady=4,
        )
        self.btn_correct.grid(row=0, column=0, padx=8)

        self.btn_wrong = tk.Button(
            feedback_frame,
            text="Wrong",
            command=self.enable_correction,
            state=tk.DISABLED,
            font=("Arial", 13),
            padx=12,
            pady=4,
        )
        self.btn_wrong.grid(row=0, column=1, padx=8)

        tk.Label(
            feedback_frame,
            text="Correct label:",
            font=("Arial", 13),
            bg="#f3f4f6",
        ).grid(row=1, column=0, padx=8, pady=10)

        self.entry_correct = tk.Entry(
            feedback_frame,
            width=8,
            justify="center",
            font=("Arial", 13),
        )
        self.entry_correct.grid(row=1, column=1, padx=8, pady=10)
        self.entry_correct.config(state=tk.DISABLED)

        self.btn_save = tk.Button(
            feedback_frame,
            text="Save fix",
            command=self.save_correction,
            state=tk.DISABLED,
            font=("Arial", 13),
            padx=10,
            pady=4,
        )
        self.btn_save.grid(row=1, column=2, padx=8, pady=10)

        self.lbl_feedback = tk.Label(
            self.main_frame,
            text="",
            font=("Arial", 14),
            bg="#f3f4f6",
        )
        self.lbl_feedback.pack(pady=(0, 4))
        self.redraw_canvas()

    def get_demo_canvas_size(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        available_height = max(480, screen_height - 420)
        available_width = max(480, screen_width - 160)
        size = min(available_width, available_height)
        cell_size = max(24, size // self.grid_size)
        return cell_size * self.grid_size

    def redraw_canvas(self):
        self.canvas.delete("all")
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                value = self.image_data[row, col]
                color = "white" if value else "black"
                self.paint_cell(row, col, color)

    def paint_cell(self, row, col, color):
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="#6b7280")

    def draw(self, event):
        self.paint(event.x, event.y, 255, "white")

    def erase(self, event):
        self.paint(event.x, event.y, 0, "black")

    def paint(self, x, y, value, color):
        col = x // self.cell_size
        row = y // self.cell_size

        if 0 <= row < self.grid_size and 0 <= col < self.grid_size:
            self.image_data[row, col] = value
            self.paint_cell(row, col, color)

    def clear(self):
        self.image_data.fill(0)
        self.redraw_canvas()
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
        self.lbl_feedback.config(text="Enter the exact label, for example 7, A, a, or ư.")

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

        feedback_path = Path(FEEDBACK_DATA_PATH)
        feedback_path.parent.mkdir(parents=True, exist_ok=True)

        if feedback_path.exists():
            feedback = np.load(feedback_path)
            X_feedback = np.concatenate([feedback["X"], new_X], axis=0)
            y_feedback = np.concatenate([feedback["y"], new_y], axis=0)
        else:
            X_feedback = new_X
            y_feedback = new_y

        np.savez_compressed(feedback_path, X=X_feedback, y=y_feedback)


if __name__ == "__main__":
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()
