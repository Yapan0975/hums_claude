# EvidLife-Map P1 Dockerfile.
#
# Base: NVIDIA CUDA 12.1 + Ubuntu 22.04 (matches Jetson Orin NX L4T 35.x runtime
# for ABI compatibility on the nvblox CUDA kernels). Python 3.10 + ROS 2 Humble
# are installed so that the nvblox bindings used by ``core/nvblox_bridge.py``
# (and the upstream ``nvblox_ros2`` package consumed in tests) can both build.
#
# Two-stage build: ``builder`` resolves apt + python deps, ``runtime`` is the
# slim image actually used for development / CI.

# ---------- stage: builder ----------
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps for ROS 2 Humble + nvblox build chain.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        gnupg2 \
        lsb-release \
        software-properties-common \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-venv \
        libglib2.0-0 \
        libgl1 \
        libxrender1 \
        libsm6 \
        libxext6 \
        libomp-dev \
        libsqlite3-dev \
        libgoogle-glog-dev \
        libgflags-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# ROS 2 Humble apt source.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-humble-ros-base \
        ros-humble-ament-cmake \
        ros-humble-rclpy \
    && rm -rf /var/lib/apt/lists/*

# Python deps — install torch from the official cu121 wheel index.
COPY requirements.txt /tmp/requirements.txt
RUN python3.10 -m pip install --upgrade pip wheel setuptools && \
    python3.10 -m pip install \
        --extra-index-url https://download.pytorch.org/whl/cu121 \
        -r /tmp/requirements.txt

# ---------- stage: runtime ----------
FROM builder AS runtime

WORKDIR /workspace
COPY . /workspace

# Editable install of the EvidLife-Map package. Tests collect even without GPU.
RUN python3.10 -m pip install -e /workspace

ENV PYTHONPATH="/workspace:${PYTHONPATH}" \
    ROS_DISTRO=humble \
    AMENT_PREFIX_PATH="/opt/ros/humble" \
    LD_LIBRARY_PATH="/opt/ros/humble/lib:${LD_LIBRARY_PATH}"

# Default: drop into a shell so the developer can run pytest / scripts manually.
SHELL ["/bin/bash", "-lc"]
CMD ["bash"]
