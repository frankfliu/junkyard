#!/usr/bin/env bash
set -exo pipefail

#############################################
# build libtiff from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/libtiff" ]]; then
        cd ${WORK_DIR}/src/libtiff
        set +e
        make clean
    fi
    exit
fi

TIFF_INSTALL_DIR=${WORK_DIR}/build/libtiff-${TIFF_VERSION}
mkdir -p ${TIFF_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "libtiff" ]; then
    git clone https://gitlab.com/libtiff/libtiff.git
fi
cd libtiff

echo "Building libtiff ${TIFF_VERSION} ..."
git fetch
git checkout Release-v${TIFF_VERSION}

./autogen.sh
./configure --quiet --disable-shared --disable-jpeg --disable-zlib --disable-jbig --disable-lzma \
    --prefix=${TIFF_INSTALL_DIR}

make -j ${NUM_PROC}
make install

tar cvfz ${TIFF_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${TIFF_INSTALL_DIR} .

cd ${WORK_DIR}
