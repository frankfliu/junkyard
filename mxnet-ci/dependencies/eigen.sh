#!/usr/bin/env bash
set -exo pipefail

#############################################
# build eigen from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/eigen" ]]; then
        cd ${WORK_DIR}/src/eigen
        rm -rf build
    fi
    exit
fi

EIGEN_INSTALL_DIR=${WORK_DIR}/build/eigen-${EIGEN_VERSION=3.3.4}
mkdir -p ${EIGEN_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "eigen" ]; then
    git clone https://github.com/eigenteam/eigen-git-mirror.git eigen
fi
cd eigen

echo "Building eigen ${EIGEN_VERSION} ..."

git fetch
git checkout ${EIGEN_VERSION}

mkdir -p build
cd build

cmake -q \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=${EIGEN_INSTALL_DIR} ..

make -j ${NUM_PROC}
make install

tar cvfz ${EIGEN_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${EIGEN_INSTALL_DIR} .

cd ${WORK_DIR}
