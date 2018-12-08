#!/usr/bin/env bash
set -exo pipefail

#############################################
# build cityhash from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/cityhash" ]]; then
        cd ${WORK_DIR}/src/cityhash
        set +e
        make clean
    fi
    exit
fi

CITYHASH_INSTALL_DIR=${WORK_DIR}/build/cityhash-${CITYHASH_VERSION}
mkdir -p ${CITYHASH_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "cityhash" ]; then
    git clone https://github.com/google/cityhash.git
fi
cd cityhash

echo "Building cityhash ${CITYHASH_VERSION} ..."

git fetch
git checkout 8af9b8c2b889d80c22d6bc26ba0df1afb79a30db

./configure -prefix=${CITYHASH_INSTALL_DIR} --enable-sse4.2

make -j ${NUM_PROC} CXXFLAGS="-g -O3 -msse4.2"
make install

tar cvfz ${CITYHASH_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${CITYHASH_INSTALL_DIR} .

cd ${WORK_DIR}
