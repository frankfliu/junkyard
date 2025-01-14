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
        pixel_values: torch.Tensor,
    ) -> dict[Any, torch.Tensor]:
        output = self.model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            pixel_values=pixel_values)
        # filter non-Tensor
        ret = OrderedDict()
        for k, v in output.items():
            if isinstance(v, torch.Tensor):
                ret[k] = v
        return ret


def main():
    model_id = "openai/clip-vit-large-patch14"
    pipe = pipeline(model=model_id, framework="pt")

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    texts = ["cat", "remote control"]

    result = pipe(image,
                  candidate_labels=texts,
                  return_tensors="pt",
                  multi_label=False)
    print(result)

    hypothesis_template = "This is a photo of {}."
    sequences = [hypothesis_template.format(x) for x in texts]

    encoding = pipe.tokenizer(sequences, padding=True, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    image_features = pipe.image_processor(images=image, return_tensors='pt')
    pixel_values = image_features["pixel_values"]

    with torch.no_grad():
        # outputs = model(**encoding)
        outputs = ModelWrapper(pipe.model)(input_ids, attention_mask,
                                           pixel_values)

    logits = outputs["logits_per_image"][0]
    probs = logits.softmax(dim=-1).numpy()

    result = [{
        "score": score,
        "label": candidate_label
    } for score, candidate_label in sorted(zip(probs, texts),
                                           key=lambda x: -x[0])]
    print(result)


def jit_trace():
    model_id = "openai/clip-vit-large-patch14"
    pipe = pipeline(model=model_id, framework="pt")

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    texts = ["cat", "remote control"]
    hypothesis_template = "This is a photo of {}."
    sequences = [hypothesis_template.format(x) for x in texts]

    encoding = pipe.tokenizer(sequences, padding=True, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    image_features = pipe.image_processor(images=image, return_tensors='pt')
    pixel_values = image_features["pixel_values"]

    traced_model = torch.jit.trace(ModelWrapper(pipe.model),
                                   (input_ids, attention_mask, pixel_values),
                                   strict=False)

    model_dir = model_id.split("/")[1]
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_model, f"{model_dir}/model.pt")


if __name__ == '__main__':
    # main()
    jit_trace()
