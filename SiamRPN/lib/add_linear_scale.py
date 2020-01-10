from torch import nn
import torch as t


class AddLinearScale(nn.Module):
    def __init__(self, std_multiplier):
        super(AddLinearScale, self).__init__()
        self.std_mutiplier = std_multiplier
        self.upper_bound = 1

    def forward(self, input):
        input_flat = input.view(-1)
        pos_input = input_flat[input_flat > 0]
        if pos_input.__len__() > 0:
            std = t._std(pos_input)
            mean = t.mean(pos_input)
            if not isinstance(std, t.cuda.FloatTensor):
                self.upper_bound = 1.0
            else:
                self.upper_bound = (mean + std * self.std_mutiplier).item()
        else:
            self.upper_bound = 1.0
        assert self.upper_bound > 0, 'upper_bound cannot be less or equal to 0'
        output = t.clamp(input, 0, self.upper_bound)
        output = output * 1.0 / self.upper_bound
        return output
