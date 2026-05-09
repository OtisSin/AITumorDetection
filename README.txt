Из-за того, что веса моодели весят слишком много, их приходится устанавливать отдельно, согласно этим командам

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