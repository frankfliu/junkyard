#!/usr/bin/env python3

import json
import os

import numpy as np
import tritonclient.http as httpclient


def np_to_triton_dtype(np_dtype):
    if np_dtype is bool:
        return "BOOL"
    elif np_dtype == np.int8:
        return "INT8"
    elif np_dtype == np.int16:
        return "INT16"
    elif np_dtype == np.int32:
        return "INT32"
    elif np_dtype == np.int64:
        return "INT64"
    elif np_dtype == np.uint8:
        return "UINT8"
    elif np_dtype == np.uint16:
        return "UINT16"
    elif np_dtype == np.uint32:
        return "UINT32"
    elif np_dtype == np.uint64:
        return "UINT64"
    elif np_dtype == np.float16:
        return "FP16"
    elif np_dtype == np.float32:
        return "FP32"
    elif np_dtype == np.float64:
        return "FP64"
    elif np_dtype == np.object_ or np_dtype.type == np.bytes_:
        return "BYTES"
    return None


def prepare_tensor(client, name, data):
    t = client.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


def main():
    kwargs = {"verbose": False}
    url = "localhost:8000"
    model_name = "fastertransformer"
    inputs = [{
        "name": "input_ids",
        "data": [[13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1]],
        "dtype": np.uint32
    }, {
        "name": "sequence_length",
        "data": [[11]],
        "dtype": np.uint32
    }, {
        "name": "max_output_len",
        "data": [[127]],
        "dtype": np.uint32
    }]
    if os.path.exists("request.json"):
        with open("request.json") as f:
            inputs = json.load(f)

    request = []
    for index, value in enumerate(inputs):
        request.append({
            "name": value["name"],
            "data": np.array(value["data"], dtype=value["dtype"]),
        })

    with httpclient.InferenceServerClient(url, **kwargs) as cl:
        request_header = [
            prepare_tensor(httpclient, field["name"], field["data"])
            for field in request
        ]

        result = cl.infer(model_name, request_header)

    for output in result.get_response()["outputs"]:
        print("{}:\n{}\n".format(output["name"],
                                 result.as_numpy(output["name"])))


if __name__ == "__main__":
    main()
