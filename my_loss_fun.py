from torch.autograd import Variable
import torch.nn as nn

loss = nn.CrossEntropyLoss()


class my_loss_fun(nn.Module):
    def __init__(self):
        super(my_loss_fun, self).__init__

    def forward(self, input, out):
        out_one = nn.L1Loss(input, out)

        return
