#!/usr/bin/env python

import json
import os

import numpy as np
import requests
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection


class _JSONEncoder(json.JSONEncoder):
    """
    custom `JSONEncoder` to make sure float and int64 ar converted
    """

    def default(self, obj):
        import datetime
        if isinstance(obj, datetime.datetime):
            return obj.__str__()

        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()

        return super(_JSONEncoder, self).default(obj)


def main():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    # VERY important: text queries need to be lowercased + end with a dot
    text = "a cat. a remote control."

    encoding = processor(images=image, text=text,
                         return_tensors="pt").to(device)
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    # token_type_ids = encoding["token_type_ids"]
    token_type_ids = None
    pixel_values = encoding["pixel_values"]
    # pixel_mask = encoding["pixel_mask"]
    pixel_mask = None

    with torch.no_grad():
        # outputs = model(**encoding)
        outputs = model(pixel_values, input_ids, token_type_ids,
                        attention_mask, pixel_mask)

    results = processor.post_process_grounded_object_detection(
        outputs,
        input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]])
    result = results[0]
    val = json.dumps(
        {
            "labels": result["labels"],
            "scores": result["scores"].tolist(),
            "boxes": result["boxes"].cpu().detach().numpy().tolist(),
        },
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        cls=_JSONEncoder,
        separators=(",", ":"))
    print(val)


def jit_trace():
    model_id = "IDEA-Research/grounding-dino-base"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    text = "a cat. a remote control."

    encoding = processor(images=image, text=text, return_tensors="pt")
    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]
    token_type_ids = encoding["token_type_ids"]
    pixel_values = encoding["pixel_values"]
    pixel_mask = encoding["pixel_mask"]

    traced_model = torch.jit.trace(
        model,
        (pixel_values, input_ids, token_type_ids, attention_mask, pixel_mask),
        strict=False)

    model_dir = model_id.split("/")[1]
    os.makedirs(model_dir, exist_ok=True)
    torch.jit.save(traced_model, f"{model_dir}/model.pt")


if __name__ == '__main__':
    main()
    # jit_trace()
