#!/usr/bin/env python


class InferInput:
    def __init__(self, name, shape, datatype):
        self._name = name
        self._shape = shape
        self._datatype = datatype
        self._parameters = {}
        self._data = None
        self._raw_data = None

    def set_data_from_numpy(self, input_tensor, binary_data=True):
        if not binary_data:
            self._raw_data = None
            self._data = [val.item() for val in input_tensor.flatten()]
        else:
            self._data = None
            self._raw_data = input_tensor.tobytes()
            self._parameters["binary_data_size"] = len(self._raw_data)

    def get_tensor(self):
        tensor = {"name": self._name, "shape": self._shape, "datatype": self._datatype}
        if self._parameters:
            tensor["parameters"] = self._parameters

        if self._raw_data is None and self._data is not None:
            tensor["data"] = self._data
        return tensor


class InferOutput:
    def __init__(self, name, binary_data=True):
        self._name = name
        self._parameters = {}
        self._binary = binary_data
        self._parameters["binary_data"] = binary_data

    def get_tensor(self):
        tensor = {"name": self._name}
        if self._parameters:
            tensor["parameters"] = self._parameters
        return tensor
