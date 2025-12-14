import yaml
from ultralytics import YOLO


def train():
    # 1. Читаем твой конфиг с настройками
    config_path = "training_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Загружена конфигурация:", config)

    # 2. Извлекаем имя весов (удаляем из конфига, так как это аргумент конструктора, а не train)
    model_weight = config.pop('model_weight', 'yolo11n.pt')

    # 3. Инициализируем модель (YOLOv8 или YOLO11 загрузятся автоматически)
    # Если yolo11n.pt не скачается сам, замени в yaml на yolov8n.pt
    model = YOLO(model_weight)

    # 4. Запускаем обучение, передавая параметры из твоего yaml файла как аргументы (**config)
    results = model.train(**config)

    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: {config.get('project', 'runs')}/{config.get('name', 'exp')}/weights/best.pt")


if __name__ == '__main__':
    train()