#!/usr/bin/env bash

set -e

PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')

if [[ "${PLATFORM}" == "linux" ]]; then
    apt-get update
    apt-get install -y --no-install-recommends \
        nasm \
        libtool \
        automake

    dependencies/openblas.sh
elif [[ "${PLATFORM}" == "darwin" ]]; then
    brew install automake pkg-config nasm cmake
fi
dependencies/zlib.sh
dependencies/libjpeg-turbo.sh
dependencies/libpng.sh
dependencies/libtiff.sh
dependencies/openssl.sh
dependencies/curl.sh
dependencies/eigen.sh
dependencies/opencv.sh
dependencies/protobuf.sh
dependencies/cityhash.sh
dependencies/libzmq.sh
dependencies/lz4.sh

