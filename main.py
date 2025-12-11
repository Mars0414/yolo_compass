import cv2
from ultralytics import YOLO

model_path = 'runs/detect/compass_run5/weights/best.pt'

model = YOLO(model_path)

names = {
    0: 'W',
    1: 'N',
    2: 'E',
    3: 'S'
}

video_path = "28446-369807704.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Ошибка открытия видео")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            conf = float(box.conf[0])

            cls = int(box.cls[0])

            label_text = names.get(cls, 'Unknown')

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{label_text} conf {conf:.2f}"
            cv2.putText(frame, text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Compass YOLO', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()