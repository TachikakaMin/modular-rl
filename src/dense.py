import numpy as np
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size,
            info,
        ):
        """
        :param output_shape: tuple containing shape of expected output
        :param input_size: size of input features
        :param info: dict containing num of hidden layers, size of hidden layers, activation function, output distribution etc.
        """
        super().__init__()
        self._output_size = output_size
        self._input_size = input_size
        self._layers = info['layers']
        self._node_size = info['node_size']
        self.activation = info['activation']
        self.dist = info['dist']
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._input_size, self._node_size)]
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, self._output_size)]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)
        return dist_inputs