#!/usr/bin/env python

from typing import List

import torch
from torch import nn, Tensor


class IValueProcessing(nn.Module):

    def __init__(self):
        super(IValueProcessing, self).__init__()

    def forward(self, input1: int, input2: int, input3: int) -> List[int]:
        # import pdb;
        # pdb.set_trace()
        return self.choose_event_times_tensor(input1, input2,
                                              input3).data.tolist()

    @torch.jit.export
    def choose_event_times_tensor(self, T: int, nevent: int,
                                  min_spacing: int) -> Tensor:
        cond = False
        times = torch.randint(low=0, high=T, size=[nevent], dtype=torch.int64)
        while not cond:
            times = torch.randint(low=0,
                                  high=T,
                                  size=[nevent],
                                  dtype=torch.int64)
            if nevent > 1:
                diffcond = False
                diffs = []
                torch_combination = torch.combinations(times, 2)
                for tensor in torch_combination:
                    item: List[int] = tensor.data.tolist()
                    x, y = item
                    diffs.append(torch.abs(torch.tensor(x - y)))
                    diffcond = bool(
                        torch.all(
                            torch.tensor([
                                diff.item() > min_spacing for diff in diffs
                            ])).item())
            else:
                diffcond = True
            # startcond = torch.all(times > 20)
            startcond = bool(
                torch.all(torch.tensor([time.item() > 20
                                        for time in times])).item())
            # endcond = torch.all(times < T - 20)
            endcond = bool(
                torch.all(
                    torch.tensor([time.item() < T - 20
                                  for time in times])).item())
            cond = diffcond and startcond and endcond
        return times


def main():
    model = IValueProcessing()
    max_events = 2
    T = 512
    min_space = T // int(max_events * 5)

    jit_processing = torch.jit.script(model)
    torch.jit.save(jit_processing, "ivalue_jit_final.pt")

    traced_model = torch.jit.load("ivalue_jit_final.pt")
    output = traced_model(T, max_events, min_space)
    print(output)


if __name__ == '__main__':
    main()
