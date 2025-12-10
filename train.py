from ultralytics import YOLO

def train():
    # Загружаем модель
    model = YOLO('yolov8n.pt')

    # Запускаем тренировку
    model.train(
        data='data.yaml',  # Наш конфиг
        epochs=5,         # 50 эпох
        imgsz=640,         # Размер картинки
        batch=8,           # Размер пакета
        name='compass_run' # Имя результата
    )

if __name__ == '__main__':
    train()