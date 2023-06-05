import torch
from torch import nn


class NeuronGeneral:
    def __init__(
        self,
        model_input_num,    # num of input from outside interface
        state_input_num,    # num of input from other neurons
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        self.model_input_num = model_input_num
        self.state_input_num = state_input_num
        self.linear = nn.Linear(model_input_num + state_input_num, 1)
        self.activation = nn.ReLU() if (activation is None) else activation
    
    def forward(self, x):
        return self.activation(self.linear(x))
