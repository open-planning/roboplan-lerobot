ARG UBUNTU_VERSION=24.04
ARG PINOCCHIO_VERSION=4.0.0

FROM ubuntu:${UBUNTU_VERSION} AS roboplan_lerobot

ARG PINOCCHIO_VERSION

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1
# Use osmesa instead for headless/CPU-only environments.
ENV MUJOCO_GL=egl
# See https://github.com/tier4/osqp_vendor/issues/26
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5

SHELL ["/bin/bash", "-c"]

RUN apt-get update -y && apt-get upgrade -y && apt-get install -y \
    cmake git build-essential pkg-config \
    libeigen3-dev libgmock-dev liburdfdom-dev libyaml-cpp-dev \
    python3-pip python3-dev \
    libegl1 libegl-mesa0 libgl1 libglfw3 libglew-dev libosmesa6-dev \
    vim nano bash-completion \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install nanobind pytest scikit-build-core

# Pinocchio via pip. The cmeel prefix must be exposed to cmake and the linker
# for the roboplan build to find it.
RUN pip3 install pin==${PINOCCHIO_VERSION}

RUN PIP_PATH="$(pip3 show pin | grep 'Location:' | cut -d' ' -f2)/cmeel.prefix" && \
    echo "export LD_LIBRARY_PATH=${PIP_PATH}/lib:\${LD_LIBRARY_PATH}" >> /roboplan_env.sh && \
    echo "export CMAKE_PREFIX_PATH=${PIP_PATH}:\${CMAKE_PREFIX_PATH}" >> /roboplan_env.sh

RUN pip3 install \
    matplotlib numpy opencv-contrib-python-headless plyfile \
    pynput tyro "viser>=1.0.29" xacro

# Build roboplan from source (no ROS required).
RUN mkdir -p /workspace
WORKDIR /workspace
COPY roboplan /workspace/roboplan

RUN source /roboplan_env.sh && \
    cd /workspace/roboplan && \
    ./scripts/build_cmake.bash

RUN echo "export CMAKE_PREFIX_PATH=/workspace/roboplan/install/:\${CMAKE_PREFIX_PATH}" \
        >> /roboplan_env.sh
RUN for pkg_lib in /workspace/roboplan/install/*/lib; do \
        echo "export LD_LIBRARY_PATH=${pkg_lib}:\${LD_LIBRARY_PATH}"; \
    done >> /roboplan_env.sh

RUN pip3 install "mujoco>=3.0.0"

# Kinova Gen3 and Robotiq 2F-85 MJCF models.
RUN git clone --depth 1 \
    https://github.com/google-deepmind/mujoco_menagerie.git \
    /opt/mujoco_menagerie
ENV MUJOCO_MENAGERIE_PATH=/opt/mujoco_menagerie

# CUDA 12.4 wheels work with driver >= 550 (Ada Lovelace and newer).
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

RUN pip3 install lerobot h5py

COPY roboplan-lerobot /workspace/roboplan-lerobot
WORKDIR /workspace/roboplan-lerobot
RUN if [ -f pyproject.toml ]; then \
        source /roboplan_env.sh && pip3 install -e .; \
    fi

COPY roboplan-lerobot/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
RUN echo "source /entrypoint.sh" >> ~/.bashrc

# Let the host user (non-root) read /root/.bashrc and /roboplan_env.sh.
RUN chmod 755 /root && chmod 644 /root/.bashrc /roboplan_env.sh
