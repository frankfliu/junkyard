#!/usr/bin/env python

import rapidjson as json
import numpy as np
from infer_input import InferInput, InferOutput


def gen_header(batch: int):
    inputs = []
    outputs = []

    input0_data = np.zeros((batch, 6, 768), dtype=np.float32)
    input0_data.tofile(f"zero_{batch}_6_768.numpy")
    input1_data = np.zeros((batch, 6), dtype=np.int32)
    input2_data = np.zeros((batch, 6), dtype=np.int32)
    input3_data = np.zeros((batch, 1), dtype=np.int32)
    input4_data = np.zeros((batch, 1), dtype=np.int32)
    input5_data = np.zeros((batch, 1), dtype=np.int32)
    input6_data = np.zeros((batch, 1), dtype=np.int32)
    input7_data = np.zeros((batch, 1), dtype=np.int32)

    inputs.append(InferInput("BERT_ENCODING__0", [batch, 6, 768], "FP32"))
    inputs.append(InferInput("ATTENTION_MASK__1", [batch, 6], "INT32"))
    inputs.append(InferInput("TOKEN_TYPES__2", [batch, 6], "INT32"))
    inputs.append(InferInput("PREV_INTENT__3", [batch, 1], "INT32"))
    inputs.append(InferInput("PREV_DA__4", [batch, 1], "INT32"))
    inputs.append(InferInput("PREV_SLOT_TO_ELICIT__5", [batch, 1], "INT32"))
    inputs.append(InferInput("ACTIVE_CONTEXT__6", [batch, 1], "INT32"))
    inputs.append(InferInput("VALID_LENGTH__7", [batch, 1], "INT32"))

    # Initialize the data
    inputs[0].set_data_from_numpy(input0_data, binary_data=True)
    inputs[1].set_data_from_numpy(input1_data, binary_data=False)
    inputs[2].set_data_from_numpy(input2_data, binary_data=False)
    inputs[3].set_data_from_numpy(input3_data, binary_data=False)
    inputs[4].set_data_from_numpy(input4_data, binary_data=False)
    inputs[5].set_data_from_numpy(input5_data, binary_data=False)
    inputs[6].set_data_from_numpy(input6_data, binary_data=False)
    inputs[7].set_data_from_numpy(input7_data, binary_data=False)

    outputs.append(InferOutput("INTENT_SCORES__0", binary_data=True))
    outputs.append(InferOutput("SLOT_SCORES__1", binary_data=True))

    infer_request = {
        "inputs": [input.get_tensor() for input in inputs],
        "outputs": [output.get_tensor() for output in outputs],
    }

    return json.dumps(infer_request)


if __name__ == "__main__":
    headers = gen_header(32)
    print(headers)
