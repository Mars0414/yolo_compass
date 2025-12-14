import cv2
import os
from ultralytics import YOLO


def main():
    # --- НАСТРОЙКИ ---
    # Путь к модели (проверь этот путь после обучения!)
    # Обычно это: my_project/yolo11n_custom/weights/best.pt
    model_path = os.path.join("my_project", "yolo11n_custom", "weights", "best.pt")

    video_path = "test_video.mp4"  # Видео для проверки (или поставь 0 для веб-камеры)
    conf_threshold = 0.5  # Порог уверенности (50%)

    # Цвета для классов (BGR формат для OpenCV)
    # North (Красный), South (Зеленый), West (Синий), East (Желтый)
    colors = {
        0: (0, 0, 255),  # North
        1: (0, 255, 0),  # South
        2: (255, 0, 0),  # West
        3: (0, 255, 255)  # East
    }
    # -----------------

    # Проверка модели
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

    # Получаем размеры видео для правильного отображения
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Видео запущено: {width}x{height}. Нажмите 'Q' для выхода.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Конец видео.")
            break

        # Инференс (распознавание)
        # verbose=False, чтобы не спамить в консоль
        results = model.predict(frame, conf=conf_threshold, verbose=False)

        # Обработка результатов
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Координаты рамки
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Класс и уверенность
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                label_name = model.names[cls_id]
                color = colors.get(cls_id, (255, 255, 255))  # Белый, если класс неизвестен

                # Формируем текст
                text = f"{label_name} {conf:.2f}"

                # Рисуем прямоугольник
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

                # Рисуем подложку для текста (чтобы читалось лучше)
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (x1, y1 - 30), (x1 + w, y1), color, -1)

                # Рисуем текст
                cv2.putText(frame, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Показываем кадр
        cv2.imshow('YOLO Compass Tracker', frame)

        # Выход на 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()