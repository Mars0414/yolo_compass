from ultralytics import YOLO

def train():

    model = YOLO('yolov8n.pt')

    model.train(
        data='data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        name='compass_run'
    )

if __name__ == '__main__':
    train()