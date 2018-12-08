#!/usr/bin/env bash
set -exo pipefail

#############################################
# build libpng from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/libpng" ]]; then
        cd ${WORK_DIR}/src/libpng
        rm -rf build
    fi
    exit
fi

PNG_INSTALL_DIR=${WORK_DIR}/build/libpng-${PNG_VERSION}
ZLIB_INSTALL_DIR=$(find ${WORK_DIR}/build -type d -name zlib-*)

mkdir -p ${PNG_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "libpng" ]; then
    git clone https://github.com/glennrp/libpng.git
fi
cd libpng

echo "Building libpng ${PNG_VERSION} ..."

git fetch
git checkout v${PNG_VERSION}

mkdir -p build
cd build

cmake -q \
    -D PNG_SHARED=OFF \
    -D PNG_STATIC=ON \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D ZLIB_LIBRARY=${ZLIB_INSTALL_DIR}/lib/libz.a \
    -D ZLIB_INCLUDE_DIR=${ZLIB_INSTALL_DIR}/include \
    -D CMAKE_INSTALL_PREFIX=${PNG_INSTALL_DIR} \
    -D CMAKE_C_FLAGS=-fPIC ..

make -j ${NUM_PROC}
make install

mkdir -p ${PNG_INSTALL_DIR}/include/libpng
ln -sf ../png.h ${PNG_INSTALL_DIR}/include/libpng/png.h

tar cvfz ${PNG_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${PNG_INSTALL_DIR} .

cd ${WORK_DIR}
