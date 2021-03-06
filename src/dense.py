import numpy as np
import torch.nn as nn
import torch.distributions as td

class DenseModel(nn.Module):
    def __init__(
            self, 
            input_size, 
            output_size,
            info,
            dropout = 0,
            init_params = None,
            zerone = False
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
        self.dropout = dropout
        self.init_params = init_params
        self.model = self.build_model(zerone)

    def build_model(self, zerone=False):
        model = [nn.Linear(self._input_size, self._node_size)]
        if self.init_params:
            nn.init.normal_(model[-1].weight, self.init_params[0], self.init_params[1])
        model += [self.activation()]
        for i in range(self._layers-1):
            model += [nn.Linear(self._node_size, self._node_size)]
            if self.init_params:
                nn.init.normal_(model[-1].weight, self.init_params[0], self.init_params[1])
            model += [nn.Dropout(self.dropout)]
            model += [self.activation()]
        model += [nn.Linear(self._node_size, self._output_size)]
        if self.init_params:
            nn.init.normal_(model[-1].weight, self.init_params[0], self.init_params[1])
        
        if zerone:
            model+= [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, input):
        dist_inputs = self.model(input)
        return dist_inputs
        # return td.independent.Independent(td.Normal(dist_inputs, 1), 1)