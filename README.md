# Proyecto Final – Control de semaforos con Visión computacional utilizando YOLOv12, un conteo dinamico y un control Heurístico

Este proyecto implementa un pipeline de visión computacional orientado a control semafórico adaptativo. Consta de tres componentes principales: (1) detección de objetos en video mediante YOLOv12, (2) conteo de vehículos por aproximación usando líneas virtuales (LineCounter) con estimación de flujos “vehículos por minuto” y desglose por clases, y (3) un controlador heurístico que decide la fase y asigna un tiempo verde dinámico por aproximación, ponderando la carga por clase (bus, camión, moto, carro, van,) y umbrales configurables. El objetivo es ofrecer una base práctica, modular y extensible para evaluar mejoras en movilidad urbana con datos visuales.

### Características clave
- **Detección**: Wrapper de Ultralytics que permite inferencia directa en tests.
- **Conteo robusto**: Cruce de líneas por seguimiento de ID; “veh_por_min” por aproximación y acumulados por clase.
- **Control heurístico**: Cálculo de demanda ponderada, umbrales (bajo/medio/alto), límite min/max de verde.
- **Ciclo semafórico**: Verde → Amarillo  → Rojo fijo, y reevaluación de fase.
- **Ejecución local y Docker**: Modo GUI en local y modo “headless” en Docker con video de salida MP4.
- **Pruebas unitarias**: Con pytest para componentes críticos como lo son la geometría, el algoritmo para el contador, el algoritmo para el detector, el controlador heurístico y un pipeline de prueba.

### Arquitectura (alto nivel)
- `src/vision/counter.py`: Lógica de conteo por intersección de segmentos, memoria de tracks, tasas de recarga.
- `src/control/cont_heuristico.py`: Para el controlador Heurístico contienen una ponderación por clase, los umbrales y un min y max del estado del semaforo en verde.
- `src/utils/*`: Utilidades geometría para calculo de los bounding box y dibujo de lineas de area objetivo.
- `final_consolidado.py`: Orquestador. Lee configuración (`config/app.yaml`), ejecuta detector+contador+control, muestra las capas y guarda video.

### Requisitos
- Python 3.10+
- `pip install -r requirements.txt`
- **Pesos del modelo**: Ubica `models/yolo12/modelo_final/weights/best.pt` (no se versiona por tamaño). se puede y se recomienda usar Git LFS.

### Instalación y ejecución (local)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:VIDEO_SOURCE="C:\ruta\a\tu_video.mp4"
python .\final_consolidado.py
