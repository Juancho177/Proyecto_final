FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender1 \
    libglib2.0-0 \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

RUN useradd -ms /bin/bash appuser
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN chown -R appuser:appuser /app

RUN mkdir -p /outputs && chown -R appuser:appuser /outputs
VOLUME ["/outputs"]

ENV HEADLESS=1
USER appuser

ENTRYPOINT ["python", "-u", "final_consolidado.py"]


