import numpy as np
from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax


class StandardCNN:
    def __init__(self, input_dim=16, num_classes=94, conv_padding=1):
        self.num_classes = num_classes
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
            Dense(input_len=flattened_size, nodes=128),
            ReLU(),
            Dense(input_len=128, nodes=num_classes),
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
        }

        if class_labels is not None:
            self.class_labels = np.asarray(class_labels)
            save_data["class_labels"] = self.class_labels

        np.savez(file_name, **save_data)
        print("Saved model weights to", file_name)

    def load_weights(self, file_name):
        data = np.load(file_name)
        self.layers[0].filters = data["conv1_filters"]
        if "conv1_biases" in data.files:
            self.layers[0].biases = data["conv1_biases"]
        self.layers[3].filters = data["conv2_filters"]
        if "conv2_biases" in data.files:
            self.layers[3].biases = data["conv2_biases"]
        self.layers[7].weights = data["dense1_weights"]
        self.layers[7].biases = data["dense1_biases"]
        self.layers[9].weights = data["dense2_weights"]
        self.layers[9].biases = data["dense2_biases"]
        self.class_labels = data["class_labels"] if "class_labels" in data.files else None
        print("Loaded model weights from", file_name)

    def predict(self, image):
        probabilities = self.forward(image)
        predicted_index = np.argmax(probabilities)
        confidence = probabilities[predicted_index] * 100
        if self.class_labels is None:
            return predicted_index, confidence
        return self.class_labels[predicted_index], confidence
