#!/usr/bin/env bash
set -ex

if [[ ! -f ".mxnet_root" ]]; then
    echo "Please change directory to mxnet source root directory."
    exit
fi

if [[ "$1" == "mkl" ]]; then
    MKL="-mkl"
fi
if [[ "${CUDA_VERSION}" != "" ]]; then
    FLAVOR="-cuda"
fi

export BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/set_env.sh

S3_PREFIX=https://s3.us-east-2.amazonaws.com/mxnet-public/precompiled_libraries/${PLATFORM}-${ARCH}

export MXNET_DIR=`pwd`
export DEPS_PATH=${MXNET_DIR}/build/deps

if [[ ! -d "${DEPS_PATH}" ]]; then
    mkdir -p ${DEPS_PATH}
    cd ${DEPS_PATH}
    if [[ $PLATFORM == 'linux' ]]; then
        curl ${S3_PREFIX}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz | tar xvz
    fi
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
	cd ${MXNET_DIR}
fi

export PKG_CONFIG_PATH=${DEPS_PATH}/lib/pkgconfig:${PKG_CONFIG_PATH}
export CPATH=${DEPS_PATH}/include:$CPATH
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${DEPS_PATH}/lib

cp -f ${BASEDIR}/config/${PLATFORM}${FLAVOR}${MKL}.mk config.mk

make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} DMLCCORE
make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} ${MXNET_DIR}/3rdparty/tvm/nnvm/lib/libnnvm.a
make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} PSLITE

if [[ "$1" == "mkl" ]]; then
    if [[ $PLATFORM == 'linux' ]]; then
        IOMP_LIBFILE='libiomp5.so'
        MKLML_LIBFILE='libmklml_intel.so'
        MKLDNN_LIBFILE='libmkldnn.so.0'
    else
        IOMP_LIBFILE='libiomp5.dylib'
        MKLML_LIBFILE='libmklml.dylib'
        MKLDNN_LIBFILE='libmkldnn.0.dylib'
    fi
    make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH} mkldnn
fi

make -j ${NUM_PROC} DEPS_PATH=${DEPS_PATH}

if [[ $PLATFORM == 'linux' ]]; then
    cp -f /usr/lib/x86_64-linux-gnu/libgfortran.so.3 lib/
    cp -f /usr/lib/x86_64-linux-gnu/libquadmath.so.0 lib/
fi
