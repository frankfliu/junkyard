#!/usr/bin/env bash

set -ex

if [[ ! "${PLATFORM}" == "darwin" ]]; then
    dependencies/openblas.sh clean
fi
dependencies/zlib.sh clean
dependencies/libjpeg-turbo.sh clean
dependencies/libpng.sh clean
dependencies/libtiff.sh clean
dependencies/openssl.sh clean
dependencies/curl.sh clean
dependencies/eigen.sh clean
dependencies/opencv.sh clean
dependencies/protobuf.sh clean
dependencies/cityhash.sh clean
dependencies/libzmq.sh clean
dependencies/lz4.sh clean
