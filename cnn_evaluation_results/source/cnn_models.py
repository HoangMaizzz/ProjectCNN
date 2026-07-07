import numpy as np
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax


class StandardCNN:
    def __init__(self, input_dim=16, num_classes=94, conv_padding=1, hidden_nodes=64):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.conv_padding = conv_padding
        self.hidden_nodes = hidden_nodes
        self.class_labels = None

        conv1_out = input_dim + (2 * conv_padding) - 3 + 1
        pool1_out = conv1_out // 2

        conv2_out = pool1_out + (2 * conv_padding) - 3 + 1
        pool2_out = conv2_out // 2

        flattened_size = pool2_out * pool2_out * 32

        self.layers = [
            Conv2D(in_channels=1, num_filters=16, filter_size=3, padding=conv_padding),
            ReLU(),
            MaxPool2D(pool_size=2),
            Conv2D(in_channels=16, num_filters=32, filter_size=3, padding=conv_padding),
            ReLU(),
            MaxPool2D(pool_size=2),
            Flatten(),
            Dense(input_len=flattened_size, nodes=hidden_nodes),
            ReLU(),
            Dense(input_len=hidden_nodes, nodes=num_classes),
            Softmax(),
        ]

    def forward(self, image):
        output = (image / 255.0) - 0.5
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, initial_gradient, learning_rate):
        gradient = initial_gradient
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient, learning_rate)

    def train_step(self, image, label, learning_rate):
        probabilities = self.forward(image)

        label = int(label)
        loss = -np.log(probabilities[label] + 1e-9)
        acc = 1 if np.argmax(probabilities) == label else 0

        gradient = probabilities.copy()
        gradient[label] -= 1

        self.backward(gradient, learning_rate)

        return loss, acc

    def save_weights(self, file_name, class_labels=None):
        save_data = {
            "conv1_filters": self.layers[0].filters,
            "conv1_biases": self.layers[0].biases,
            "conv2_filters": self.layers[3].filters,
            "conv2_biases": self.layers[3].biases,
            "dense1_weights": self.layers[7].weights,
            "dense1_biases": self.layers[7].biases,
            "dense2_weights": self.layers[9].weights,
            "dense2_biases": self.layers[9].biases,
            "input_dim": np.array(self.input_dim, dtype=np.int32),
            "num_classes": np.array(self.num_classes, dtype=np.int32),
            "conv_padding": np.array(self.conv_padding, dtype=np.int32),
            "hidden_nodes": np.array(self.hidden_nodes, dtype=np.int32),
        }

        if class_labels is not None:
            self.class_labels = np.asarray(class_labels)
            save_data["class_labels"] = self.class_labels

        np.savez(file_name, **save_data)
        print("Saved model weights to", file_name)

    def load_weights(self, file_name):
        with np.load(file_name, allow_pickle=False) as data:
            loaded = {
                "conv1_filters": np.array(data["conv1_filters"], copy=True),
                "conv2_filters": np.array(data["conv2_filters"], copy=True),
                "dense1_weights": np.array(data["dense1_weights"], copy=True),
                "dense1_biases": np.array(data["dense1_biases"], copy=True),
                "dense2_weights": np.array(data["dense2_weights"], copy=True),
                "dense2_biases": np.array(data["dense2_biases"], copy=True),
            }
            expected_shapes = {
                "conv1_filters": self.layers[0].filters.shape,
                "conv2_filters": self.layers[3].filters.shape,
                "dense1_weights": self.layers[7].weights.shape,
                "dense1_biases": self.layers[7].biases.shape,
                "dense2_weights": self.layers[9].weights.shape,
                "dense2_biases": self.layers[9].biases.shape,
            }
            for name, expected_shape in expected_shapes.items():
                if loaded[name].shape != expected_shape:
                    raise ValueError(
                        f"Model shape mismatch for {name}: expected {expected_shape}, "
                        f"found {loaded[name].shape}. Retrain with the current architecture."
                    )

            self.layers[0].filters = loaded["conv1_filters"]
            if "conv1_biases" in data.files:
                self.layers[0].biases = np.array(data["conv1_biases"], copy=True)
            self.layers[3].filters = loaded["conv2_filters"]
            if "conv2_biases" in data.files:
                self.layers[3].biases = np.array(data["conv2_biases"], copy=True)
            self.layers[7].weights = loaded["dense1_weights"]
            self.layers[7].biases = loaded["dense1_biases"]
            self.layers[9].weights = loaded["dense2_weights"]
            self.layers[9].biases = loaded["dense2_biases"]
            self.class_labels = (
                np.array(data["class_labels"], copy=True)
                if "class_labels" in data.files else None
            )
        print("Loaded model weights from", file_name)

    def predict(self, image):
        probabilities = self.forward(image)
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index] * 100
        if self.class_labels is None:
            return predicted_index, confidence
        return self.class_labels[predicted_index], confidence
