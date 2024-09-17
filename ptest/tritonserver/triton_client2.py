#!/usr/bin/env python3

import json
import os
import struct

import numpy as np
import requests


def np_to_triton_dtype(np_dtype):
    if np_dtype == bool:
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

def triton_to_npdtype(dtype):
    if dtype == "BOOL":
        return bool
    elif dtype == "INT8":
        return np.int8
    elif dtype == "INT16":
        return np.int16
    elif dtype == "INT32":
        return np.int32
    elif dtype == "INT64":
        return np.int64
    elif dtype == "UINT8":
        return np.uint8
    elif dtype == "UINT16":
        return np.uint16
    elif dtype == "UINT32":
        return np.uint32
    elif dtype == "UINT64":
        return np.uint64
    elif dtype == "FP16":
        return np.float16
    elif dtype == "FP32":
        return np.float32
    elif dtype == "FP64":
        return np.float64
    elif dtype == "BYTES":
        return np.bytes_
    return None

def prepare_tensor(client, name, data):
    t = client.InferInput(name, data.shape, np_to_triton_dtype(data.dtype))
    t.set_data_from_numpy(data)
    return t


def main():
    kwargs = {"verbose": False}
    url = "localhost:8000"
    model_name = "fastertransformer"
    inputs = [
        {
            "name": "input_ids",
            "data": [[13959, 1566, 12, 2968, 10, 37, 629, 19, 1627, 5, 1]],
            "dtype": np.uint32
        },
        {
            "name": "sequence_length",
            "data": [[11]],
            "dtype": np.uint32
        },
        {
            "name": "max_output_len",
            "data": [[127]],
            "dtype": np.uint32
        }
    ]
    if os.path.exists("request.json"):
        with open("request.json") as f:
            inputs = json.load(f)

    json_data = {
        "inputs": [],
        "parameters": {
            "binary_data_output": True
        }
    }

    binary_data = b''
    for value in inputs:
        array = np.array(value["data"], dtype=value["dtype"])
        buf = array.tobytes()
        binary_data += buf
        json_data["inputs"].append({
            "name": value["name"],
            "shape": array.shape,
            "datatype": np_to_triton_dtype(array.dtype),
            "parameters": {
                "binary_data_size": len(buf)
            }
        })

    json_request = json.dumps(json_data)
    with open("header.json", "w") as f:
        f.write(json_request)

    headers = {
        "Inference-Header-Content-Length": str(len(json_request))
    }
    request_body = struct.pack(f"{len(json_request)}s{len(binary_data)}s", json_request.encode(), binary_data)
    with open("payload.bin", "wb") as f:
        f.write(request_body)

    print(f"curl -f -X POST --data-binary @payload.bin"
          f" -H 'Inference-Header-Content-Length: {len(json_request)}'"
          f" http://localhost:8000/v2/models/{model_name}/infer")

    request_uri = f"http://localhost:8000/v2/models/{model_name}/infer"
    result = requests.post(request_uri, headers=headers, data=request_body)
    json_size = int(result.headers["Inference-Header-Content-Length"])
    json_response = json.loads(result.content[:json_size].decode())


    offset = json_size
    for out in json_response["outputs"]:
        size = out["parameters"]["binary_data_size"]
        array = np.frombuffer(result.content[offset:offset + size], dtype=triton_to_npdtype(out["datatype"]))
        array = array.reshape(out["shape"])
        offset += size
        print(f"{out['name']} ({array.dtype} {array.shape}):\n{array}")

if __name__ == "__main__":
    main()
