#!/usr/bin/env bash
set -ex

. ./set_env.sh

ARCH=$(uname -m)
S3_PREFIX=https://s3.us-east-2.amazonaws.com/mxnet-public/precompiled_libraries/${PLATFORM}-${ARCH}

export DEPS_PATH=${WORK_DIR}/build/deps

if [[ ! -d "${DEPS_PATH}" ]]; then
    mkdir -p ${DEPS_PATH}
    cd ${DEPS_PATH}
    curl ${S3_PREFIX}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/zlib-${ZLIB_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/libjpeg-turbo-${TURBO_JPEG_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/libpng-${PNG_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/libtiff-${TIFF_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/openssl-${OPENSSL_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/curl-${LIBCURL_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/eigen-${EIGEN_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/opencv-${OPENCV_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/protobuf-${PROTOBUF_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/cityhash-${CITYHASH_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/libzmq-${ZEROMQ_VERSION}.tar.gz | tar xvz
    curl ${S3_PREFIX}/lz4-${LZ4_VERSION}.tar.gz | tar xvz
fi

cd ${WORK_DIR}

export PKG_CONFIG_PATH=$DEPS_PATH/lib/pkgconfig:$PKG_CONFIG_PATH
export CPATH=$DEPS_PATH/include:$CPATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:$DEPS_PATH/lib

make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} DMLCCORE
make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} ${WORK_DIR}/3rdparty/tvm/nnvm/lib/libnnvm.a
make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} PSLITE

make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH}
