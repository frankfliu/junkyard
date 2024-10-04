#!/usr/bin/env python
import tensorflow as tf
import tensorflow.keras as keras


def main():
    loaded_model = keras.models.load_model(
        "/Users/lufen/Downloads/deepdanbooru-v3-20211112-sgd-e28/model-resnet_custom_v3.h5"
    )
    tf.saved_model.save(loaded_model, "resnet/1/")


if __name__ == '__main__':
    main()
