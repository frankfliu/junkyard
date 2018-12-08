#!/usr/bin/env bash
set -exo pipefail

#############################################
# build libjpeg-turbo from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/libjpeg-turbo" ]]; then
        cd ${WORK_DIR}/src/libjpeg-turbo
        rm -rf build
    fi
    exit
fi

TURBO_JPEG_INSTALL_DIR=${WORK_DIR}/build/libjpeg-turbo-${TURBO_JPEG_VERSION}
mkdir -p ${TURBO_JPEG_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "libjpeg-turbo" ]; then
    git clone https://github.com/libjpeg-turbo/libjpeg-turbo.git
fi
cd libjpeg-turbo

echo "Building libjpeg-turbo ${TURBO_JPEG_VERSION} ..."

git fetch
git checkout ${TURBO_JPEG_VERSION}

mkdir -p build
cd build

if [[ "${PLATFORM}" == "darwin" ]]; then
    JPEG_NASM_OPTION="-D CMAKE_ASM_NASM_COMPILER=/usr/local/bin/nasm"
fi

cmake -q \
    -G"Unix Makefiles" \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${TURBO_JPEG_INSTALL_DIR} \
    -D CMAKE_C_FLAGS=-fPIC \
    -D WITH_JAVA=FALSE \
    -D WITH_JPEG7=TRUE \
    -D WITH_JPEG8=TRUE \
    $JPEG_NASM_OPTION \
    -D ENABLE_SHARED=FALSE ..

make -j ${NUM_PROC}
make install

tar cvfz ${TURBO_JPEG_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${TURBO_JPEG_INSTALL_DIR} .

cd ${WORK_DIR}
