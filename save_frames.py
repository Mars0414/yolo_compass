import cv2
import os
import numpy as np

video_paths = ["gettyimages-816687434-640_adpp.mp4",
               "gettyimages-953461144-640_adpp.mp4",
               "gettyimages-1224551772-640_adpp.mp4",
               "28446-369807704.mp4"]
for video_path in video_paths:
    prefix = video_path.split(".")[0]
    output_folder = 'frames'
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

        if count % step == 0:
            filename = os.path.join(output_folder, f'frame_{prefix}_{count}.jpg')
            cv2.imwrite(filename, frame)
            saved_count += 1
            print(f'Saved frame {saved_count} (index {count})')

        count += 1

    cap.release()
    print("Готово!")