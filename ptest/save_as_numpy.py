#!/usr/bin/env python

import io

import numpy as np


def save_npy(file: str, array):
    np.save(file, array)


def save_npz(file: str, arrays):
    np.savez(file, arrays)


def save_raw(file: str, array):
    array.tofile(file)


def to_raw_bytes(array):
    return array.tobytes()


if __name__ == "__main__":
    data = [np.zeros((1, 2)), np.ones((1, 128))]

    # Save single array into .npy file
    save_npy("zero_1_2.npy", data[0])

    # Save multiple arrays into .npz file
    save_npz("zero_1_2.npz", *data)

    # Save single array raw data into file
    save_raw("zero_1_2.bin", data[0])

    out = np.frombuffer(data[0].tobytes())

    memory_file = io.BytesIO()
    np.savez(memory_file, *data)
    memory_file.seek(0)

    t = memory_file.getbuffer().tobytes()
    b = bytearray(t)
    if b[0] == 80 and b[1] == 75:
        buf = io.BytesIO(b)
        npz = np.load(buf)
        out = []
        for item in npz.items():
            out.append(item[1])

    f = open("img_bytes.bin", "wb")
    f.write(data[0].tobytes())
    f.close()
