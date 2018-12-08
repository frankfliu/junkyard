#!/usr/bin/env bash
set -exo pipefail

#############################################
# build protobuf from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/protobuf" ]]; then
        cd ${WORK_DIR}/src/protobuf
        set +e
        make clean
    fi
    exit
fi

PROTOBUF_INSTALL_DIR=${WORK_DIR}/build/protobuf-${PROTOBUF_VERSION}
mkdir -p ${PROTOBUF_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "protobuf" ]; then
    git clone https://github.com/protocolbuffers/protobuf.git
fi
cd protobuf

echo "Building protobuf ${PROTOBUF_VERSION} ..."

git fetch
git checkout v${PROTOBUF_VERSION}

./autogen.sh
./configure -prefix=${PROTOBUF_INSTALL_DIR}

make -j ${NUM_PROC}
make install

tar cvfz ${PROTOBUF_INSTALL_DIR}.tar.gz -C ${PROTOBUF_INSTALL_DIR} .

cd ${WORK_DIR}
