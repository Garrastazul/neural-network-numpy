import numpy as np




class LayerDense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = np.random.rand(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        self.output = np.dot(self.weights, inputs) + self.biases

class Activation_ReLU():
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activation_Softmax():
    def forward(self, inputs):
        e_z = np.exp(inputs - np.max(inputs))
        self.output = e_z / np.sum(e_z, axis=1, keepdims=True)