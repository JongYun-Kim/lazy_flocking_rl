# syntax=docker/dockerfile:1.6
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

# system
RUN sed -i -E 's#http://(archive|security)\.ubuntu\.com/ubuntu#https://\1.ubuntu.com/ubuntu#g' /etc/apt/sources.list \
 && printf 'Acquire::Retries "5";\nAcquire::http::Timeout "10";\nAcquire::https::Timeout "10";\n' > /etc/apt/apt.conf.d/99net-tuning \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    software-properties-common \
    ca-certificates curl wget git build-essential \
    sudo nano x11-apps ncdu \
    iproute2 iputils-ping net-tools dnsutils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 ffmpeg \
    gnupg \
 && rm -rf /var/lib/apt/lists/*

# install glow from Charm apt repo
RUN mkdir -p /etc/apt/keyrings \
 && curl -fsSL https://repo.charm.sh/apt/gpg.key \
    | gpg --dearmor -o /etc/apt/keyrings/charm.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/charm.gpg] https://repo.charm.sh/apt/ * *" \
    > /etc/apt/sources.list.d/charm.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends glow \
 && rm -rf /var/lib/apt/lists/*

# python
RUN add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    python3.9 python3.9-distutils python3.9-venv \
 && rm -rf /var/lib/apt/lists/*

# fix python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
 && /usr/bin/python3.9 /tmp/get-pip.py \
 && rm -f /tmp/get-pip.py \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip \
 && ln -sf /usr/local/bin/pip3.9 /usr/local/bin/pip3

# double check
RUN python -V && python3 -V && pip -V

# upgrade pip tooling
RUN python -m pip install --upgrade pip setuptools wheel

# PyTorch cu113
RUN python -m pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu113 \
    torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1+cu113

# install remaining python dependencies from docker context
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt

# user config
ARG UID=1006
ARG GID=1006
ARG USERNAME=flocking
ARG PASSWORD=lazy

RUN groupadd -g ${GID} ${USERNAME} || true \
 && useradd -m -s /bin/bash -u ${UID} -g ${GID} ${USERNAME} \
 && echo "${USERNAME}:${PASSWORD}" | chpasswd \
 && mkdir -p /home/${USERNAME}/.ssh \
 && chown -R ${UID}:${GID} /home/${USERNAME}/.ssh \
 && chmod 700 /home/${USERNAME}/.ssh \
 && echo "${USERNAME} ALL=(ALL) ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME} \
 && printf '\nexport PATH="$HOME/.local/bin:$PATH"\nexport LANG=C.UTF-8\nexport LC_ALL=C.UTF-8\n' >> /home/${USERNAME}/.bashrc \
 && chown ${UID}:${GID} /home/${USERNAME}/.bashrc

RUN mkdir -p /workspace \
 && chown -R ${UID}:${GID} /workspace

WORKDIR /workspace

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER ${USERNAME}

CMD ["/bin/bash"]
