import numpy as np

class Conv2D:
    def __init__(self, in_channels, num_filters, filter_size=3, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.filters = np.random.randn(num_filters, filter_size, filter_size, in_channels) * np.sqrt(2.0 / (in_channels * filter_size * filter_size))
        self.biases = np.zeros(num_filters)

    def iterate_regions(self, image):
        h, w = image.shape[0], image.shape[1]
        for i in range(h - self.filter_size + 1):
            for j in range(w - self.filter_size + 1):
                yield image[i:(i + self.filter_size), j:(j + self.filter_size)], i, j

    def forward(self, input_data):
        if len(input_data.shape) == 2:
            input_data = np.expand_dims(input_data, axis=2)

        self.last_input = input_data
        if self.padding > 0:
            input_data = np.pad(
                input_data,
                ((self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode='constant'
            )

        self.last_padded_input = input_data
        h, w = input_data.shape[0], input_data.shape[1]
        output = np.zeros((h - self.filter_size + 1, w - self.filter_size + 1, self.num_filters))

        for im_region, i, j in self.iterate_regions(input_data):
            for f in range(self.num_filters):
                output[i, j, f] = np.sum(im_region * self.filters[f]) + self.biases[f]
        return output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_filters = np.zeros(self.filters.shape)
        d_L_d_biases = np.zeros(self.biases.shape)
        d_L_d_input = np.zeros(self.last_padded_input.shape)

        for im_region, i, j in self.iterate_regions(self.last_padded_input):
            for f in range(self.num_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * im_region
                d_L_d_biases[f] += d_L_d_out[i, j, f]
                d_L_d_input[i:i+self.filter_size, j:j+self.filter_size] += d_L_d_out[i, j, f] * self.filters[f]

        self.filters -= learning_rate * d_L_d_filters
        self.biases -= learning_rate * d_L_d_biases

        if self.padding > 0:
            return d_L_d_input[
                self.padding:-self.padding,
                self.padding:-self.padding,
                :
            ]
        return d_L_d_input

class ReLU:
    def forward(self, input_data):
        self.last_input = input_data
        return np.maximum(0, input_data)

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_input = np.copy(d_L_d_out)
        d_L_d_input[self.last_input <= 0] = 0
        return d_L_d_input

class MaxPool2D:
    def __init__(self, pool_size=2):
        self.pool_size = pool_size

    def iterate_regions(self, image):
        h, w, num_filters = image.shape
        new_h, new_w = h // self.pool_size, w // self.pool_size
        for i in range(new_h):
            for j in range(new_w):
                yield image[(i * self.pool_size):(i * self.pool_size + self.pool_size), 
                            (j * self.pool_size):(j * self.pool_size + self.pool_size)], i, j

    def forward(self, input_data):
        self.last_input = input_data
        h, w, num_filters = input_data.shape
        output = np.zeros((h // self.pool_size, w // self.pool_size, num_filters))
        for im_region, i, j in self.iterate_regions(input_data):
            output[i, j] = np.amax(im_region, axis=(0, 1))
        return output

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_input = np.zeros(self.last_input.shape)
        for im_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = im_region.shape
            amax = np.amax(im_region, axis=(0, 1))
            
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if im_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * self.pool_size + i2, j * self.pool_size + j2, f2] += d_L_d_out[i, j, f2]
        return d_L_d_input

class Flatten:
    def forward(self, input_data):
        self.last_input_shape = input_data.shape
        return input_data.flatten()

    def backward(self, d_L_d_out, learning_rate):
        return d_L_d_out.reshape(self.last_input_shape)

class Dense:
    def __init__(self, input_len, nodes):
        self.weights = np.random.randn(input_len, nodes) * np.sqrt(2.0 / input_len)
        self.biases = np.zeros(nodes)

    def forward(self, input_data):
        self.last_input = input_data
        return np.dot(input_data, self.weights) + self.biases

    def backward(self, d_L_d_out, learning_rate):
        d_L_d_weights = np.outer(self.last_input, d_L_d_out)
        d_L_d_input = np.dot(d_L_d_out, self.weights.T)
        
        self.weights -= learning_rate * d_L_d_weights
        self.biases -= learning_rate * d_L_d_out
        
        return d_L_d_input

class Softmax:
    def forward(self, input_data):
        self.last_totals = input_data
        tmp = np.exp(input_data - np.max(input_data))
        self.last_output = tmp / np.sum(tmp)
        return self.last_output

    def backward(self, d_L_d_out, learning_rate):
        return d_L_d_out
