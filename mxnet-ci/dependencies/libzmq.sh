#!/usr/bin/env bash
set -exo pipefail

#############################################
# build libzmq from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/libzmq" ]]; then
        cd ${WORK_DIR}/src/libzmq
        rm -rf build
    fi
    exit
fi

ZEROMQ_INSTALL_DIR=${WORK_DIR}/build/libzmq-${ZEROMQ_VERSION}
mkdir -p ${ZEROMQ_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "libzmq" ]; then
    git clone https://github.com/zeromq/libzmq.git
fi
cd libzmq

echo "Building libzmq ${ZEROMQ_VERSION} ..."

git fetch
git checkout v${ZEROMQ_VERSION}

mkdir -p build
cd build

cmake -q \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${ZEROMQ_INSTALL_DIR} \
    -D WITH_LIBSODIUM=OFF \
    -D BUILD_SHARED_LIBS=OFF ..

make -j ${NUM_PROC}
make install

tar cvfz ${ZEROMQ_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${ZEROMQ_INSTALL_DIR} .

cd ${WORK_DIR}
