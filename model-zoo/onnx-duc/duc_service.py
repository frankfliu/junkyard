import base64
import json
import math
import os
import sys
from os import path

import cv2 as cv
import mxnet as mx
import numpy as np
from PIL import Image
from mxnet.io import DataBatch

from model_handler import ModelHandler


def get_palette():
    sys.path.append(path.dirname(path.abspath(__file__)))
    import cityscapes_labels
    # get train id to color mappings from file
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette


def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert("P")
    result_img.putpalette(get_palette())
    return np.array(result_img.convert("RGB"))


class DUCService(ModelHandler):

    def __init__(self):
        super(DUCService, self).__init__()
        self.mxnet_ctx = None
        self.mx_model = None
        self.labels = None
        self.signature = None
        self.epoch = 0
        self.input_shape = None

    # noinspection PyMethodMayBeStatic
    def get_model_files_prefix(self, context):
        return context.manifest["model"]["modelName"]

    def initialize(self, context):
        super(DUCService, self).initialize(context)

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        gpu_id = properties.get("gpu_id")
        batch_size = properties.get("batch_size")
        if batch_size > 1:
            raise ValueError("Batch is not supported.")

        signature_file_path = os.path.join(model_dir, "signature.json")
        if not os.path.isfile(signature_file_path):
            raise RuntimeError("Missing signature.json file.")

        with open(signature_file_path) as f:
            self.signature = json.load(f)

        self.mxnet_ctx = mx.cpu() if gpu_id is None else mx.gpu(gpu_id)

    def preprocess(self, batch):
        properties = self._context.system_properties
        model_dir = properties.get("model_dir")
        model_files_prefix = self.get_model_files_prefix(self._context)
        checkpoint_prefix = "{}/{}".format(model_dir, model_files_prefix)

        img_list = []
        for idx, data in enumerate(batch):
            img = data.get("data")
            if img is None:
                img = data.get("body")

            img_arr = mx.img.imdecode(img, to_rgb=0).asnumpy()
            self.input_shape = img_arr.shape[:-1]
            # load model with input shape
            sym, arg_params, aux_params = mx.model.load_checkpoint(checkpoint_prefix, self.epoch)
            self.mx_model = mx.mod.Module(symbol=sym, context=self.mxnet_ctx, data_names=["data"], label_names=None)
            self.mx_model.bind(for_training=False,
                               data_shapes=[("data", (1, 3, self.input_shape[0], self.input_shape[1]))])
            self.mx_model.set_params(arg_params, aux_params, allow_missing=True, allow_extra=True)
            # set rgb mean of input image (used in mean subtraction)
            rgb_mean = cv.mean(img_arr)
            # Convert to float32
            test_img = img_arr.astype(np.float32)
            # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
            test_shape = self.input_shape
            cell_shapes = [math.ceil(l / 8) * 8 for l in test_shape]
            test_img = cv.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - test_shape[0]), 0,
                                         max(0, int(cell_shapes[1]) - test_shape[1]), cv.BORDER_CONSTANT,
                                         value=rgb_mean)
            test_img = np.transpose(test_img, (2, 0, 1))
            # subtract rbg mean
            for i in range(3):
                test_img[i] -= rgb_mean[i]
            test_img = np.expand_dims(test_img, axis=0)
            # convert to ndarray
            test_img = mx.ndarray.array(test_img)
            img_list.append(test_img)
        return img_list

    def inference(self, model_input):
        model_input = [item.as_in_context(self.mxnet_ctx) for item in model_input]
        self.mx_model.forward(DataBatch(model_input))
        model_input = self.mx_model.get_outputs()
        # by pass lazy evaluation get_outputs either returns a list of nd arrays
        # a list of list of ndarrays
        for d in model_input:
            if isinstance(d, list):
                for n in model_input:
                    if isinstance(n, mx.ndarray.ndarray.NDArray):
                        n.wait_to_read()
            elif isinstance(d, mx.ndarray.ndarray.NDArray):
                d.wait_to_read()
        return model_input

    def postprocess(self, inference_output):
        # get input and output dimensions
        result_height, result_width = self.input_shape
        img_height, img_width = self.input_shape
        # set downsampling rate
        ds_rate = 8
        # set cell width
        cell_width = 2
        # number of output label classes
        label_num = 19

        labels = inference_output[0].asnumpy().squeeze()
        # re-arrange output
        test_width = int((int(img_width) / ds_rate) * ds_rate)
        test_height = int((int(img_height) / ds_rate) * ds_rate)
        feat_width = int(test_width / ds_rate)
        feat_height = int(test_height / ds_rate)
        labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
        labels = np.transpose(labels, (0, 3, 1, 4, 2))
        labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

        labels = labels[:, :int(img_height / cell_width), :int(img_width / cell_width)]
        labels = np.transpose(labels, [1, 2, 0])
        labels = cv.resize(labels, (result_width, result_height), interpolation=cv.INTER_LINEAR)
        labels = np.transpose(labels, [2, 0, 1])

        # get softmax output
        softmax = labels

        # get classification labels
        results = np.argmax(labels, axis=0).astype(np.uint8)
        raw_labels = results

        # comput confidence score
        confidence = float(np.max(softmax, axis=0).mean())

        # generate segmented image
        result_img = np.reshape(colorize(raw_labels), (result_height, result_width, 3))
        result_string = base64.b64encode(result_img)

        return [{'confidence': str(confidence), 'image': result_string}]


_service = DUCService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
