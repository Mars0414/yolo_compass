import cv2
import os
import numpy as np

video_path = "28446-369807704.mp4"
output_folder = 'frames_output'
step = 10

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)
count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Если это нужный кадр (каждый 10-й)
    if count % step == 0:
        filename = os.path.join(output_folder, f'frame_{count}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1
        print(f'Saved frame {saved_count} (index {count})')

    # Считаем ВСЕ кадры, а не только сохраненные
    count += 1

# Закрываем видео ТОЛЬКО когда цикл закончился
cap.release()
print("Готово!")
