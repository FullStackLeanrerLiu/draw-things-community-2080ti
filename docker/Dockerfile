# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# Draw Things gRPC server build (ccv SM75 fork) in an official-matching
# Ubuntu 22.04 environment.
#
# WHY: the build box runs Ubuntu 24.04 (gcc 13 + python 3.12 + glibc 2.39),
# while the official publishing environment is Ubuntu 22.04
# (gcc 11 + python 3.10 + glibc 2.35). That mismatch breaks three things:
#   1. SwiftLog 1.4.4  : `localtime` returns `_Nullable` under glibc 2.39
#                       -> Swift imports it as Optional -> compile error.
#                       (fixed by pinning SwiftLog 1.5.4 in WORKSPACE.linux)
#   2. rules_cuda      : crosstool_wrapper python emits DeprecationWarning
#                       on python 3.12; bazel treats any stderr as failure.
#   3. SwiftNIO C code : `_Atomic _Bool` + `__atomic_fetch_add` is rejected
#                       by gcc 13 but accepted by gcc 11.
#
# This image reproduces the official Ubuntu 22.04 toolchain so the ccv SM75
# (FlashAttention fp16 + INT8-QK) changes can be compiled and verified.
#
# Build (run on the build box):
#   docker build -t dt-grpc-build-22.04 .
#
# Run (mount the checked-out workspace with the gitmirror + bazel cache):
#   docker run --rm -it \
#     --gpus '"device=0"' \
#     -v /home/tom/draw-things-community-2080ti:/src \
#     -v /home/tom/gitmirror:/home/tom/gitmirror \
#     -v /home/tom/.cache/bazel:/home/tom/.cache/bazel \
#     -e http_proxy=http://127.0.0.1:7897 \
#     -e https_proxy=http://127.0.0.1:7897 \
#     --network host \
#     dt-grpc-build-22.04 \
#     bash -lc "cd /src && ./build_linux.sh"
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

# ---- base toolchain (matches official 22.04: gcc11 / py3.10 / glibc2.35) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git pkg-config make cmake \
      gcc g++ python3 python3-dev libpython3-dev \
      zlib1g-dev libicu-dev \
      openjdk-21-jdk-headless \
    && rm -rf /var/lib/apt/lists/*

# rules_cuda finds `python`; ensure python3.10 is also reachable as `python`.
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10 \
    || ln -s /usr/bin/python3 /usr/bin/python

# ---- Bazelisk (pins bazel to the version the workspace expects) ----
ENV BAZELISK_HOME=/opt/bazelisk
RUN curl -fsSL -o /usr/local/bin/bazel \
      https://github.com/bazelbuild/bazelisk/releases/latest/download/bazelisk-linux-amd64 \
    && chmod +x /usr/local/bin/bazel

# ---- Swift toolchain 6.0.3 (same as official base image) ----
ENV SWIFT_VERSION=6.0.3 SWIFT_WORKAROUND_UBUNTU=ubuntu2204
ARG SWIFT_URL=https://download.swift.org/swift-${SWIFT_VERSION}-release/${SWIFT_WORKAROUND_UBUNTU}/swift-${SWIFT_VERSION}-RELEASE/swift-${SWIFT_VERSION}-RELEASE-ubuntu22.04.tar.gz
RUN curl -fsSL -o /tmp/swift.tar.gz "${SWIFT_URL}" \
    && mkdir -p /opt/swift \
    && tar -xzf /tmp/swift.tar.gz -C /opt/swift --strip-components=1 \
    && rm -rf /tmp/swift.tar.gz
ENV PATH="/opt/swift/usr/bin:${PATH}"

# Python warnings from rules_cuda + crosstool wrapper must be suppressed so
# bazel (which treats stderr as failure) does not abort. python 3.10 still
# emits them for these legacy scripts.
ENV PYTHONWARNINGS="ignore::SyntaxWarning:ignore::DeprecationWarning:ignore::DeprecationWarning"

WORKDIR /src
CMD ["/bin/bash"]