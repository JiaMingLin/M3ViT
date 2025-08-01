FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"

# Install system dependencies and Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common curl wget git unzip && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    autoconf automake libtool \
    zlib1g-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libfreetype6-dev \
    liblcms2-dev \
    libwebp-dev \
    libopenjp2-7-dev \
    libharfbuzz-dev \
    libfribidi-dev && \
    rm -rf /var/lib/apt/lists/*

# Install pip
RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py

# Create symlinks
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    ln -s $(which pip3) /usr/bin/pip

# === 安裝 Python 套件，並避免 pip cache 打包進 image ===
RUN pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir \
    imageio \
    scikit-image \
    pyyaml \
    easydict \
    termcolor \
    h5py \
    mmcv==1.7.1 \
    thop \
    dm-tree \
    timm==0.6.13 \
    Pillow==6.1 \
    opencv-python==4.7.0.72 && \
    pip install --no-cache-dir "numpy<2" && \
    # 額外刪除 pip 暫存與 http cache
    rm -rf /root/.cache/pip /root/.cache/http /root/.cache/* || true

# Download and extract LibTorch 2.1.2+cu121
RUN wget https://download.pytorch.org/libtorch/test/cu121/libtorch-shared-with-deps-2.1.2%2Bcu121.zip -O libtorch.zip && \
    unzip libtorch.zip -d /opt && \
    rm libtorch.zip

ENV LIBTORCH_PATH=/opt/libtorch
ENV CPLUS_INCLUDE_PATH=$LIBTORCH_PATH/include:$LIBTORCH_PATH/include/torch/csrc:$LIBTORCH_PATH/include/torch/csrc/distributed

# Set workdir and install FastMoE
WORKDIR /app
RUN git clone https://github.com/JiaMingLin/fastmoe.git && \
    cd fastmoe && \
    git checkout pytorch2-compat && \
    python setup.py install && \
    cd .. && rm -rf /root/.cache/pip /root/.cache/http /root/.cache/* || true

# Entrypoint
RUN echo '#!/bin/bash\n' \
    'echo "==> Listing /app contents:"\n' \
    'ls -alh /app\n' \
    'echo ""\n' \
    'if [[ ! -x "/app/start.sh" ]]; then\n' \
    '  echo "❌ ERROR: start.sh not found or not executable in /app!" >&2\n' \
    '  exit 1\n' \
    'fi\n' \
    'echo "🚀 Launching /app/start.sh..."\n' \
    '/app/start.sh' > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
