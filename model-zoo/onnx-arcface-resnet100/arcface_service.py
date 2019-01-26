# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#     http://www.apache.org/licenses/LICENSE-2.0
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
ArcFaceService
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os

import cv2
import mxnet as mx
import numpy as np
from sklearn import preprocessing
from mtcnn_detector import MtcnnDetector
from mxnet_model_service import MXNetModelService
from skimage import transform as trans


class ArcFaceService(MXNetModelService):

    def preprocess(self, batch):
        mtcnn_path = os.path.dirname(__file__)
        det_threshold = [0.6, 0.7, 0.8]
        detector = MtcnnDetector(model_folder=mtcnn_path,
                                 ctx=self.mxnet_ctx,
                                 num_worker=1,
                                 accurate_landmark=True,
                                 threshold=det_threshold)

        img_list = [batch[0].get("img1"), batch[0].get("img2")]
        face_list = []
        for img in img_list:
            if img is None:
                self.error = "Requires two image input."
                return None

            img_arr = mx.img.imdecode(img, to_rgb=0).asnumpy()
            img_arr = get_input(detector, img_arr)
            if img_arr is None:
                self.error = "No face detected."
                return None

            img_arr = mx.nd.array(img_arr)
            img_arr = img_arr.expand_dims(axis=0)
            face_list.append([img_arr])

        return face_list

    def inference(self, data):
        ret = []
        for face in data:
            ret.append(copy.deepcopy(super(ArcFaceService, self).inference(face)))

        return ret

    def postprocess(self, data):
        f1 = data[0][0].asnumpy()
        f1 = preprocessing.normalize(f1).flatten()

        f2 = data[1][0].asnumpy()
        f2 = preprocessing.normalize(f2).flatten()

        dist = np.linalg.norm(f1 - f2).item()
        sim = np.dot(f1, f2.T).tolist()
        return [{"Distance": dist, "Similarity": sim}]


def pre_process(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

    tform = trans.SimilarityTransform()
    tform.estimate(dst, src)
    M = tform.params[0:2, :]

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped


def get_input(detector, face_img):
    ret = detector.detect_face(face_img, det_type=0)
    if ret is None:
        return None

    bbox, points = ret
    if bbox.shape[0] == 0:
        return None

    bbox = bbox[0, 0:4]
    points = points[0, :].reshape((2, 5)).T
    nimg = pre_process(face_img, bbox, points, image_size='112,112')
    nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
    aligned = np.transpose(nimg, (2, 0, 1))
    return aligned


_service = ArcFaceService()


def handle(data, context):
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
