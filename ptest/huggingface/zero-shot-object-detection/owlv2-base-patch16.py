#!/usr/bin/env python
import os
from typing import Any
from collections import OrderedDict

import requests
import torch
from PIL import Image
from torch import nn
from transformers import pipeline
from transformers.modeling_outputs import BaseModelOutput


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
    model_id = "google/owlv2-base-patch16"
    pipe = pipeline(model=model_id, framework="pt", device="cpu")

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # texts = ["a cat", "a remote control"]
    texts = ["a cat"]

    result = pipe(image=image, candidate_labels=texts, return_tensors="pt")
    print(result)

    encoding = pipe.tokenizer(texts, padding=True, return_tensors='pt')
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    image_features = pipe.image_processor(images=image, return_tensors='pt')
    pixel_values = image_features["pixel_values"]

    with torch.no_grad():
        # outputs = model(**encoding)
        outputs = ModelWrapper(pipe.model)(input_ids, attention_mask,
                                           pixel_values)

    target_sizes = torch.Tensor([pixel_values.shape[-2:]])

    model_outputs = BaseModelOutput(outputs)
    results = pipe.image_processor.post_process_object_detection(
        outputs=model_outputs, target_sizes=target_sizes)

    i = 0  # Retrieve predictions for the first image for the corresponding text queries
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[
        i]["labels"]

    for box, score, label in zip(boxes, scores, labels):
        box = [round(i, 2) for i in box.tolist()]
        print(
            f"Detected cat with confidence {round(score.item(), 3)} at location {box}"
        )


def jit_trace():
    model_id = "google/owlv2-base-patch16"
    pipe = pipeline(model=model_id, framework="pt", device="cpu")

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # texts = ["a cat", "a remote control"]
    texts = ["a cat"]

    encoding = pipe.tokenizer(texts, padding=True, return_tensors='pt')
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

    pipe.tokenizer.save_pretrained(model_dir)


if __name__ == '__main__':
    # main()
    jit_trace()
