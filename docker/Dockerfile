FROM python:3.10-slim

# Dependencias de sistema (OpenCV, ffmpeg, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsm6 libxext6 libxrender1 libglib2.0-0 libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Crear usuario no root
RUN useradd -ms /bin/bash appuser
WORKDIR /app

# Instalar dependencias Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código del proyecto
COPY . /app
RUN chown -R appuser:appuser /app

# Directorio de outputs
RUN mkdir -p /outputs && chown -R appuser:appuser /outputs
VOLUME ["/outputs"]

# Variables de entorno
ENV HEADLESS=1
USER appuser

# Healthcheck sigue apuntando a tu script
HEALTHCHECK --interval=30s --timeout=10s --retries=3 CMD python healthcheck.py || exit 1

# Exponer puerto de Streamlit
EXPOSE 8501

# === Entrada por defecto: levantar Streamlit ===
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

