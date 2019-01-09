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

if [[ "${PLATFORM}" == "darwin" ]]; then
    CONFIG_FLAG="--with-darwinssl --without-libidn2"
fi
./buildconf
./configure $CONFIG_FLAG \
    --with-zlib=${WORK_DIR}/build/zlib-${ZLIB_VERSION} \
    --without-nghttp2 \
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

tar cvfz ${LIBCURL_INSTALL_DIR}.tar.gz --exclude="./bin" -C ${LIBCURL_INSTALL_DIR} .

cd ${WORK_DIR}
