#!/usr/bin/env bash

NUM_PROC=1
if [[ ! -z $(command -v nproc) ]]; then
    NUM_PROC=$(nproc)
elif [[ ! -z $(command -v sysctl) ]]; then
    NUM_PROC=$(sysctl -n hw.ncpu)
fi
export NUM_PROC

export WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PLATFORM=$(uname | tr '[:upper:]' '[:lower:]')
export ARCH=$(uname -m)

export CC="gcc -fPIC"
export CXX="g++ -fPIC"
export FC="gfortran"

export OPENBLAS_VERSION=0.3.5
export ZLIB_VERSION=1.2.6
export TURBO_JPEG_VERSION=1.5.90
export PNG_VERSION=1.6.34
export TIFF_VERSION="4-0-9"
export OPENSSL_VERSION=1_0_2l
export LIBCURL_VERSION=7_61_0
export EIGEN_VERSION=3.3.4
export OPENCV_VERSION=3.4.2
export PROTOBUF_VERSION=3.5.1
export CITYHASH_VERSION=1.1.1
export ZEROMQ_VERSION=4.2.2
export LZ4_VERSION=r130

mkdir -p ${WORK_DIR}/src
