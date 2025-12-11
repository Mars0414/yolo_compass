import os
import shutil
import random

images_source_dir = "frames_output"
labels_source_dir = "labes_raw"
dataset_dir = "dataset"
train_ratio = 0.8

def split_files():
    if not os.path.exists(images_source_dir):
        print(f"ОШИБКА: Папка {images_source_dir} не найдена!")
        return

    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    for split in ['train', 'val']:
        for dtype in ['images', 'labels']:
            os.makedirs(os.path.join(dataset_dir, split, dtype), exist_ok=True)

    images = [f for f in os.listdir(images_source_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if not images:
        print(f"ОШИБКА: В папке {images_source_dir} нет картинок!")
        return

    random.shuffle(images)
    train_count = int(len(images) * train_ratio)

    print(f"Найдено картинок: {len(images)}")
    print(f"Копируем {train_count} в train и {len(images) - train_count} в val...")

    copied_count = 0
    for i, image_file in enumerate(images):
        split = 'train' if i < train_count else 'val'

        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"

        src_image = os.path.join(images_source_dir, image_file)
        src_label = os.path.join(labels_source_dir, label_file)

        if not os.path.exists(src_label):
            src_label = os.path.join(images_source_dir, label_file)
            if not os.path.exists(src_label):
                print(f"Пропуск: нет txt файла для {image_file}")
                continue

        dst_image = os.path.join(dataset_dir, split, 'images', image_file)
        dst_label = os.path.join(dataset_dir, split, 'labels', label_file)

        shutil.copy(src_image, dst_image)
        shutil.copy(src_label, dst_label)
        copied_count += 1

    print(f"\nУСПЕХ! Скопировано {copied_count} пар файлов.")
    print(f"Теперь запустите python main.py")


if __name__ == "__main__":
    split_files()