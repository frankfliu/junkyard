#!/usr/bin/env bash
set -exo pipefail

#############################################
# build curl from source
#############################################
BASEDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. ${BASEDIR}/../set_env.sh

if [[ "$1" == "clean" ]]; then
    if [[ -d "${WORK_DIR}/src/curl" ]]; then
        cd ${WORK_DIR}/src/curl
        set +e
        make clean
    fi
    exit
fi

LIBCURL_INSTALL_DIR=${WORK_DIR}/build/curl-${LIBCURL_VERSION}
mkdir -p ${LIBCURL_INSTALL_DIR}
cd ${WORK_DIR}/src/

if [ ! -d "curl" ]; then
    git clone https://github.com/curl/curl.git
fi
cd curl

echo "Building curl ${LIBCURL_VERSION} ..."

git fetch
git checkout curl-${LIBCURL_VERSION}

if [[ "${PLATFORM}" == "linux" ]]; then
    CONFIG_FLAG=""
    OLD_PKG_CONFIG_PATH=${PKG_CONFIG_PATH}
    export PKG_CONFIG_PATH=${LIBCURL_INSTALL_DIR}/lib/pkgconfig
elif [[ "${PLATFORM}" == "darwin" ]]; then
    CONFIG_FLAG="--with-darwinssl"
fi
./buildconf
./configure $CONFIG_FLAG \
    --with-zlib \
    --with-nghttps2 \
    --without-zsh-functions-dir \
    --without-librtmp \
    --without-libssh2 \
    --disable-debug \
    --disable-curldebug \
    --enable-symbol-hiding=yes \
    --enable-optimize=yes \
    --enable-shared=no \
    --enable-http=yes \
    --enable-ipv6=yes \
    --disable-ftp \
    --disable-ldap \
    --disable-ldaps \
    --disable-rtsp \
    --disable-proxy \
    --disable-dict \
    --disable-telnet \
    --disable-tftp \
    --disable-pop3 \
    --disable-imap \
    --disable-smb \
    --disable-smtp \
    --disable-gopher \
    --disable-manual \
    --prefix=${LIBCURL_INSTALL_DIR}

make -j ${NUM_PROC}
make install

if [[ "${PLATFORM}" == "linux" ]]; then
    export PKG_CONFIG_PATH=${OLD_PKG_CONFIG_PATH}
fi

tar cvfz ${LIBCURL_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${LIBCURL_INSTALL_DIR} .

cd ${WORK_DIR}
