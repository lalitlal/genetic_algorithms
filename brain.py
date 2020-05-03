import numpy as np


class Brain():

    def __init__(self, chromosome):
        super(Brain, self).__init__()

        self.architecture = [
            {'input_dim': 4, 'output_dim': 10, 'activation': Brain.relu},
            {'input_dim': 10, 'output_dim': 4, 'activation': Brain.sigmoid}
        ]

        self.chromosome = chromosome
        self.params_values = {
            'W1': chromosome['weights.1'], 'b1': chromosome['bias.1'],
            'W2': chromosome['weights.2'], 'b2': chromosome['bias.2']
        }

    def sigmoid(Z):
        return 1 / (1 + np.exp(-Z))

    def relu(Z):
        return np.maximum(0, Z)

    def forward(self, X):
        data = np.array(X)[:, np.newaxis]
        for idx, layer in enumerate(self.architecture):
            layer_idx = idx + 1
            activ_function_curr = layer["activation"]
            W = self.params_values["W" + str(layer_idx)]
            b = self.params_values["b" + str(layer_idx)]
            data = activ_function_curr(np.matmul(W, data) + b)
        return list(data.T[0])
