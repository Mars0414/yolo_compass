import os
import shutil
import random

source_images = "frames"
source_labels = "labels"
dataset_dir = "dataset"
train_ratio = 0.8

def split_data():
    if not os.path.exists(source_images) or not os.path.exists(source_labels):
        print("ОШИБКА: Не найдена папка 'frames' или 'labels'!")
        print("Убедитесь, что вы запускаете скрипт из папки yolo_compass")
        return

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    images = [f for f in os.listdir(source_images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    random.seed(42)
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    print(f"Найдено картинок: {len(images)}")
    print(f"Разбиение: {train_count} в train, {len(images) - train_count} в val")

    missing_labels = 0
    for i, image_file in enumerate(images):
        split = 'train' if i < train_count else 'val'

        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"

        src_img_path = os.path.join(source_images, image_file)
        src_lbl_path = os.path.join(source_labels, label_file)

        if not os.path.exists(src_lbl_path):
            print(f"ВНИМАНИЕ: Нет файла разметки для {image_file}. Пропуск.")
            missing_labels += 1
            continue

        dst_img_path = os.path.join(dataset_dir, 'images', split, image_file)
        dst_lbl_path = os.path.join(dataset_dir, 'labels', split, label_file)

        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_lbl_path, dst_lbl_path)

    print("\nГОТОВО!")
    if missing_labels > 0:
        print(f"Пропущено файлов без разметки: {missing_labels}")
    print(f"Папка '{dataset_dir}' создана и заполнена.")


if __name__ == "__main__":
    split_data()