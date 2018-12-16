#!/usr/bin/env bash
set -exo pipefail

#############################################
# build zlib from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/zlib" ]]; then
        cd ${WORK_DIR}/src/zlib
        rm -rf build
    fi
    exit
fi

ZLIB_INSTALL_DIR=${WORK_DIR}/build/zlib-${ZLIB_VERSION}
mkdir -p ${ZLIB_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "zlib" ]; then
    git clone https://github.com/LuaDist/zlib.git
fi
cd zlib

echo "Building zlib ${ZLIB_VERSION} ..."

git fetch
git checkout ${ZLIB_VERSION}

rm -rf build
mkdir -p build
cd build

cmake -q \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${ZLIB_INSTALL_DIR} \
    -D BUILD_SHARED_LIBS=OFF ..

make -j ${NUM_PROC}
make install

tar cvfz ${ZLIB_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${ZLIB_INSTALL_DIR} .

cd ${WORK_DIR}
