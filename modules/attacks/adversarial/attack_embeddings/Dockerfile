FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y git libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY . .

CMD []