#!/usr/bin/env python

import json

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

        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass

        return super(_JSONEncoder, self).default(obj)


def main():
    model_id = "IDEA-Research/grounding-dino-base"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    image_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(image_url, stream=True).raw)
    # Check for cats and remote controls
    # VERY important: text queries need to be lowercased + end with a dot
    text = "a cat. a remote control."

    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]]
    )
    result = results[0]
    val = json.dumps({
        "labels": result["labels"],
        "scores": result["scores"].tolist(),
        "boxes": result["boxes"].cpu().detach().numpy().tolist(),
    },
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        cls=_JSONEncoder,
        separators=(",", ":")).encode("utf-8")
    print(val)


if __name__ == '__main__':
    main()
