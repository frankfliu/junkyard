#!/usr/bin/env bash
set -exo pipefail

#############################################
# build lz4 from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/lz4" ]]; then
        cd ${WORK_DIR}/src/lz4
        set +e
        make clean
    fi
    exit
fi

LZ4_INSTALL_DIR=${WORK_DIR}/build/lz4-${LZ4_VERSION}
mkdir -p ${LZ4_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "lz4" ]; then
    git clone https://github.com/lz4/lz4.git
fi
cd lz4

echo "Building lz4 ${LZ4_VERSION} ..."

git fetch
git checkout ${LZ4_VERSION}

make -j ${NUM_PROC} CXXFLAGS="-g -O3 -msse4.2"
make PREFIX=${LZ4_INSTALL_DIR} install

tar cvfz ${LZ4_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${LZ4_INSTALL_DIR} .

cd ${WORK_DIR}
