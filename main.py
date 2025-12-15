import cv2
import os
from ultralytics import YOLO


def main():
    model_path = os.path.join("my_project", "yolo11n_custom", "weights", "best.pt")

    video_path = "test_video.mp4"
    conf_threshold = 0.5

    colors = {
        0: (255, 0, 0),
        1: (0, 255, 0),
        2: (0, 255, 255),
        3: (0, 0, 255)
    }
    if not os.path.exists(model_path):
        print(f"ОШИБКА: Модель не найдена по пути: {model_path}")
        print("Сначала запустите train.py и дождитесь окончания обучения.")
        return

    print(f"Загрузка модели: {model_path}...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ОШИБКА: Не удалось открыть видео {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Видео запущено: {width}x{height}. Нажмите 'Q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Конец видео.")
            break

        results = model.predict(frame, conf=conf_threshold, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label_name = model.names[cls_id]
                color = colors.get(cls_id, (255, 255, 255))

                text = f"{label_name} {conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)

                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        cv2.imshow('YOLO Compass Tracker', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()