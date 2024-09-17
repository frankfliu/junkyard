#!/usr/bin/env python
#
# Copyright 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.

import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-tiny", device=device)

    img = Image.open("truck.jpg")
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    input_box = np.array([425, 600, 700, 875])

    with torch.inference_mode():
        predictor.set_image(img)
        masks, scores, logits = predictor.predict(input_point, input_label, input_box)
        print(f"masks: {masks}")
        print(f"scores: {scores}")
        # print(f"logits: {logits}")


if __name__ == '__main__':
    main()
