# Детектор опухолей мозга по МРТ-снимкам

Веб-приложение для автоматического обнаружения и локализации опухолей на МРТ-снимках. 
Работает на базе Faster R-CNN (ResNet-50 FPN), контейнеризировано через Docker и управляется через интерфейс Gradio.
Проект настроен для запуска одной командой. Убедитесь, что на компьютере установлен Docker Desktop.

```bash
docker compose up --build

Не скачивайте репозиторий как ZIP-архив! В архиве вместо весов окажется текстовый указатель (~130 байт), и модель не запустится.

# 1. Установите Git LFS (если ещё нет)
sudo apt update
sudo apt install git-lfs
git lfs install

# 2. Клонируйте репозиторий
git clone https://github.com/OtisSin/AITumorDetection.git

# 3. Перейдите в папку
cd AITumorDetection

# 4. Проверьте, что файл весов на месте и имеет правильный размер
ls -lh best_faster_rcnn.pth
# Должно показать: ~150M (если покажет 130B — выполните: git lfs pull)

# 5. Запустите проект
docker compose up -d --build

# Технологии
Python 3.10
PyTorch + TorchVision
Gradio (веб-интерфейс)
OpenCV, NumPy
Docker & Docker Compose
Git LFS