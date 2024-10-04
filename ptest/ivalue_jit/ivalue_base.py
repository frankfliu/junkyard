from typing import List

import torch
from torch import nn


class IValueProcessing(nn.Module):

    def __init__(self):
        super(IValueProcessing, self).__init__()

    def forward(self, tokenized_data: List[int], _cls_token: int,
                _sep_token: int):
        output_token_ids = [_cls_token] + tokenized_data + [_sep_token]
        return output_token_ids


jit_processing = torch.jit.script(IValueProcessing())
torch.jit.save(jit_processing, "ivalue_jit.pt")
