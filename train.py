import yaml
from ultralytics import YOLO


def train():
    config_path = "training_config.yaml"

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("Загружена конфигурация:", config)

    model_weight = config.pop('model_weight', 'yolo11n.pt')

    model = YOLO(model_weight)

    results = model.train(**config)

    print("Обучение завершено!")
    print(f"Лучшая модель сохранена в: {config.get('project', 'runs')}/{config.get('name', 'exp')}/weights/best.pt")


if __name__ == '__main__':
    train()