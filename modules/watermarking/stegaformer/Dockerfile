FROM python:3.10-slim

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

RUN pip install timm==0.6.2.dev0

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/model

CMD ["python", "run_watermark.py", "--input_dir", "/data/in", "--output_dir", "/data/out", "--model", "model/stegaformer.pth"]

