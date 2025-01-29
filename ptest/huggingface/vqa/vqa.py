#!/usr/bin/env python
import os
from collections import OrderedDict
from typing import Any

import requests
import torch
from PIL import Image
from torch import nn
from transformers import pipeline


class ModelWrapper(nn.Module):

    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        pixel_mask: torch.Tensor,
    ) -> dict[Any, torch.Tensor]:
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
        )
        # filter non-Tensor
        ret = OrderedDict()
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v
        return ret


def main(model_id: str):
    pipe = pipeline("visual-question-answering", model=model_id, device="cpu")

    question = "how many dogs are in the picture?"
    img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'
    image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

    encoding = pipe.tokenizer([question], padding=True, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]

    image_features = pipe.image_processor(images=image, return_tensors='pt')
    pixel_values = image_features["pixel_values"]
    pixel_mask = image_features["pixel_mask"]

    result = pipe(image, question, top_k=1)
    print(result)

    traced_model = torch.jit.trace(
        ModelWrapper(pipe.model),
        (input_ids, attention_mask, token_type_ids, pixel_values, pixel_mask),
        strict=False)

    model_dir = model_id.split("/")[1]
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_model, f"{model_dir}/model.pt")

    pipe.tokenizer.save_pretrained(model_dir)


if __name__ == '__main__':
    # model_id = "Salesforce/blip-vqa-base"
    # model_id = "google/deplot"
    main("dandelin/vilt-b32-finetuned-vqa")
