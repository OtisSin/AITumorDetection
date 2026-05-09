FROM python:3.10-slim

WORKDIR /app

#  Устанавливаем необходимые системные библиотеки для headless OpenCV
# libglib2.0-0, libsm6, libxext6, libxrender1, libgomp1 — достаточно для работы cv2.imread/resize/imwrite
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем зависимости и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и веса
COPY AI_App.py .
COPY AI_Detect.py .
COPY best_faster_rcnn.pth .

EXPOSE 7860

CMD ["python", "AI_App.py"]