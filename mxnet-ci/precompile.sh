#!/usr/bin/env bash

set -e

apt-get update
apt-get install -y --no-install-recommends \
    nasm \
    libtool \
    automake

if [[ ! "${PLATFORM}" == "darwin" ]]; then
    dependencies/openblas.sh
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

