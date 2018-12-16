#!/usr/bin/env bash
set -e

export DEBIAN_FRONTEND=noninteractive
export DEBCONF_NONINTERACTIVE_SEEN=true
apt-get update
apt-get install -y --no-install-recommends \
		fakeroot \
		ca-certificates \
		dpkg-dev \
		g++-4.8 \
		gcc-4.8 \
		gfortran-4.8 \
		pkg-config \
		cmake \
		curl \
		wget \
		vim \
		git \
		unzip \
		python3-dev
apt-get clean
rm -rf /var/lib/apt/lists/*
cd /tmp
curl -O https://bootstrap.pypa.io/get-pip.py
python3 get-pip.py
rm get-pip.py

update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-4.8 50 --slave /usr/bin/g++ g++ /usr/bin/g++-4.8
update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-4.8 50

mkdir -p /usr/local/mxnet/config
