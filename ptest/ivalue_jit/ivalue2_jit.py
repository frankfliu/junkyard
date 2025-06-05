import torch
from torch import nn


class IValueProcessing(nn.Module):
    def __init__(self):
        super(IValueProcessing, self).__init__()

    def forward(self, tokenized_data):
        return tokenized_data


jit_processing = torch.jit.script(IValueProcessing())
torch.jit.save(jit_processing, "ivalue2_jit.pt")
