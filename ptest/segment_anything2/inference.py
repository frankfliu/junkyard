import json

import cv2
import numpy as np

from sam2_onnx import SegmentAnything2ONNX


def str2bool(v):
    return v.lower() in ("true", "1")


def main():
    model = SegmentAnything2ONNX(
        "/Users/frankliu/source/models/encoder.onnx",
        "/Users/frankliu/source/models/sam2-hiera-tiny.onnx",
    )

    image = cv2.imread("images/truck.jpg")
    prompt = json.load(open("images/truck_prompt.json"))

    embedding = model.encode(image)

    masks = model.predict_masks(embedding, prompt)

    # Merge masks
    mask = np.zeros((masks.shape[2], masks.shape[3], 3), dtype=np.uint8)
    for m in masks[0, :, :, :]:
        mask[m > 0.5] = [255, 0, 0]


if __name__ == '__main__':
    main()
