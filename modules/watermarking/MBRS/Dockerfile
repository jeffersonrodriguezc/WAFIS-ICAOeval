FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

WORKDIR /app

COPY . .

RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -f /etc/apt/keyrings/cuda-archive-keyring.gpg

RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg2 curl ca-certificates && \
    mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub | gpg --dearmor -o /etc/apt/keyrings/nvidia-cuda-keyring.gpg && \
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/nvidia-cuda-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r requirements.txt

CMD ["tail", "-f", "/dev/null"]
