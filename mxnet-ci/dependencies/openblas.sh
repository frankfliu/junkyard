#!/usr/bin/env bash
set -exo pipefail

#############################################
# build OpenBLAS from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/OpenBLAS" ]]; then
        cd ${WORK_DIR}/src/OpenBLAS
        set +e
        make clean
    fi
    exit
fi

OPENBLAS_INSTALL_DIR=${WORK_DIR}/build/OpenBLAS-${OPENBLAS_VERSION}
mkdir -p ${OPENBLAS_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "OpenBLAS" ]; then
    git clone https://github.com/xianyi/OpenBLAS.git
fi
cd OpenBLAS

echo "Building OpenBLAS ${OPENBLAS_VERSION} ..."

git fetch
git checkout v${OPENBLAS_VERSION}

make -j ${NUM_PROC} DYNAMIC_ARCH=1 NO_SHARED=1 USE_OPENMP=1

set +e
make PREFIX=${OPENBLAS_INSTALL_DIR} install
set -e

ln -sf libopenblas.a ${OPENBLAS_INSTALL_DIR}/lib/libcblas.a
ln -sf libopenblas.a ${OPENBLAS_INSTALL_DIR}/lib/liblapack.a

cp -f lapack-netlib/LAPACKE/include/lapacke*.h ${OPENBLAS_INSTALL_DIR}/include

tar cvfz ${OPENBLAS_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${OPENBLAS_INSTALL_DIR} .

cd ${WORK_DIR}
