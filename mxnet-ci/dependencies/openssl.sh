#!/usr/bin/env bash
set -exo pipefail

#############################################
# build openssl from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/openssl" ]]; then
        cd ${WORK_DIR}/src/openssl
        set +e
        make clean
    fi
    exit
fi

OPENSSL_INSTALL_DIR=${WORK_DIR}/build/openssl-${OPENSSL_VERSION}
mkdir -p ${OPENSSL_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "openssl" ]; then
    git clone https://github.com/openssl/openssl.git
fi
cd openssl

echo "Building openssl ${OPENSSL_VERSION} ..."

git fetch
git checkout OpenSSL_${OPENSSL_VERSION}

if [[ "${PLATFORM}" == "linux" ]]; then
    TARGET=linux-x86_64
elif [[ "${PLATFORM}" == "darwin" ]]; then
    TARGET=darwin64-x86_64-cc
fi
./Configure no-shared no-zlib --prefix=${OPENSSL_INSTALL_DIR} --openssldir=${WORK_DIR}/build/ssl ${TARGET}

make -j ${NUM_PROC}
make install

tar cvfz ${OPENSSL_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${OPENSSL_INSTALL_DIR} .

cd ${WORK_DIR}
