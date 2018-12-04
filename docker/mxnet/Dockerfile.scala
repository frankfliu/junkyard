FROM mxnet-cpu

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
            openjdk-8-jdk-headless \
            maven \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

