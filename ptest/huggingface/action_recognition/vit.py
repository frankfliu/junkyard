#!/usr/bin/env python

# import PIL
import torch
from transformers import pipeline
from PIL import Image
import requests


def main():
    pipe = pipeline("image-classification", "rvv-karma/Human-Action-Recognition-VIT-Base-patch16-224")
    url = "https://images.pexels.com/photos/175658/pexels-photo-175658.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500"
    image = Image.open(requests.get(url, stream=True).raw)

    _out = pipe(image)

    # image = Image.open(image)
    # image = PIL.ImageOps.exif_transpose(image)
    # image = image.convert("RGB")

    # print(_out)

    inputs = torch.zeros((1, 3, 224, 224))
    pipe.model.config.return_dict = False
    _array = pipe.model(inputs)
    traced_model = torch.jit.trace(pipe.model, inputs)
    array2 = traced_model(inputs)
    torch.jit.save(traced_model, "vit.pt")
    print(array2[0].shape)


if __name__ == "__main__":
    main()
