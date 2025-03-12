# 📌 Проверить, в какой директории ты сейчас находишься
pwd  

# 📌 Посмотреть список файлов и папок в текущей директории
ls  # Для Linux/macOS
dir # Для Windows PowerShell

# 📌 Перейти в нужную папку (например, в src/model_training)
cd src/model_training  

# 📌 Подняться на уровень выше (если случайно зашел не туда)
cd .. 

# 📌 Вернуться в предыдущую директорию (где был до этого)
cd -  

# 📌 Очистить терминал (чтобы не было мусора на экране)
clear  # Для Linux/macOS
cls    # Для Windows PowerShell

# 📌 Активировать виртуальное окружение (если используешь conda)
conda activate mlops_env  

# 📌 Активировать venv (если окружение на venv)
source venv/bin/activate  # Для Linux/macOS
venv\Scripts\activate     # Для Windows

# 📌 Проверить, какие библиотеки установлены в окружении
pip list   
conda list  # Если используешь conda

# 📌 Установить зависимости из requirements.txt (если разворачиваешь проект)
pip install -r requirements.txt  

# 📌 Запустить Python-скрипт (например, run.py
python run.py  

# 📌 Запустить Python в интерактивном режиме (для быстрых тестов)
python  

# 📌 Выйти из Python-интерактивного режима
exit()  # или Ctrl + Z и Enter в Windows, Ctrl + D в Linux/macOS  

# 📌 Проверить, какие процессы Python сейчас запущены
ps aux | grep python  # Для Linux/macOS
tasklist | findstr python  # Для Windows

# 📌 Проверить, какие процессы нагружают GPU (если используешь CUDA)
nvidia-smi  

# 📌 Запустить скрипт и записывать весь вывод в файл (для отладки)
python run.py > output.log 2>&1  

# 📌 Открыть VS Code в текущей папке (если запускаешь из терминала)
code .  

# 📌 Удалить все неиспользуемые файлы и папки (если нужно почистить проект)
rm -rf __pycache__/ *.log  # Удаляет кеш и логи





# 📌 Запуск MLflow Server с PostgreSQL в качестве бэкенда для хранения данных
mlflow server `
    --backend-store-uri postgresql://mlflow_user:mlflow_pass123@localhost/mlflow_db `
    --default-artifact-root ./mlflow_artifacts `
    --host 0.0.0.0 --port 5000

# 📌 Пояснение:
# --backend-store-uri - указывает, где хранить данные (здесь PostgreSQL)
# --default-artifact-root - путь, где сохранять артефакты моделей
# --host 0.0.0.0 - разрешает подключения извне
# --port 5000 - задает порт для доступа







MLOPS_PROJECT/
│── config/                # Конфигурационные файлы
│   ├── sweep_config.yaml  # W&B Sweep settings
│   ├── mlflow_config.yaml # MLflow settings (если используешь)
│── data/                  # Данные проекта
│   ├── raw/               # Исходные данные
│   │   ├── housing.csv
│   ├── processed/         # Предобработанные данные
│   │   ├── housing_cleaned.csv
│   │   ├── housing_scaled_minmax.csv
│   │   ├── housing_scaled_standard.csv
│── models/                # Обученные модели
│   ├── best_xgb_model.json
│── notebooks/             # Jupyter ноутбуки для исследований
│── src/                   # Основной код проекта
│   ├── data/              # Модули работы с данными
│   │   ├── download_data.py
│   │   ├── clean_data.py
│   ├── features/          # Feature Engineering
│   │   ├── feature_engineering.py
│   ├── training/          # Модули обучения моделей
│   │   ├── train.py
│   │   ├── train_sweep.py
│   │   ├── model_training.py
│   ├── evaluation/        # Оценка моделей
│   │   ├── evaluate.py
│   ├── deployment/        # Развертывание моделей (API, CI/CD)
│   │   ├── model_deployment.py
│── sweeps/                # W&B Sweeps + Hyperparameter Tuning
│   ├── sweep.py
│   ├── train_sweep.py
│── tracking/              # Monitoring & Experiment Tracking
│   ├── wandb/             # W&B Logs
│   ├── mlflow/            # MLflow Runs
│── wandb/                 # W&B project cache
│── .gitignore             # Игнорируемые файлы
│── environment.yaml       # Описание окружения (conda/pip)
│── requirements.txt       # Зависимости (если pip)
│── README.md              # Описание проекта