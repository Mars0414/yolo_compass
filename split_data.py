import os
import shutil
import random

# --- НАСТРОЙКИ (Под ваш скриншот) ---
source_images = "frames"  # Ваша папка с исходными картинками
source_labels = "labels"  # Ваша папка с исходными txt файлами
dataset_dir = "dataset"  # Имя папки, которая создастся (куда всё разложим)
train_ratio = 0.8  # 80% файлов на обучение, 20% на проверку


# -----------------------------

def split_data():
    # 1. Проверка, существуют ли исходные папки
    if not os.path.exists(source_images) or not os.path.exists(source_labels):
        print("ОШИБКА: Не найдена папка 'frames' или 'labels'!")
        print("Убедитесь, что вы запускаете скрипт из папки yolo_compass")
        return

    # 2. Очистка старой папки dataset, если она была, чтобы не было дублей
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)

    # 3. Создаем структуру папок (images/train, labels/train и т.д.)
    for split in ['train', 'val']:
        os.makedirs(os.path.join(dataset_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, 'labels', split), exist_ok=True)

    # 4. Получаем список картинок
    images = [f for f in os.listdir(source_images) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

    # Перемешиваем случайно
    random.seed(42)  # Фиксируем случайность, чтобы при повторном запуске разбиение было тем же
    random.shuffle(images)

    train_count = int(len(images) * train_ratio)

    print(f"Найдено картинок: {len(images)}")
    print(f"Разбиение: {train_count} в train, {len(images) - train_count} в val")

    # 5. Раскидываем файлы
    missing_labels = 0
    for i, image_file in enumerate(images):
        # Определяем, куда кидать (train или val)
        split = 'train' if i < train_count else 'val'

        # Имена файлов
        base_name = os.path.splitext(image_file)[0]
        label_file = base_name + ".txt"

        # Полные пути к исходникам
        src_img_path = os.path.join(source_images, image_file)
        src_lbl_path = os.path.join(source_labels, label_file)

        # Проверяем, есть ли txt файл для этой картинки
        if not os.path.exists(src_lbl_path):
            # Если txt файла нет, пропускаем картинку (или можно кинуть без метки, но для обучения это плохо)
            print(f"ВНИМАНИЕ: Нет файла разметки для {image_file}. Пропуск.")
            missing_labels += 1
            continue

        # Куда копируем
        dst_img_path = os.path.join(dataset_dir, 'images', split, image_file)
        dst_lbl_path = os.path.join(dataset_dir, 'labels', split, label_file)

        # Копируем
        shutil.copy(src_img_path, dst_img_path)
        shutil.copy(src_lbl_path, dst_lbl_path)

    print("\nГОТОВО!")
    if missing_labels > 0:
        print(f"Пропущено файлов без разметки: {missing_labels}")
    print(f"Папка '{dataset_dir}' создана и заполнена.")


if __name__ == "__main__":
    split_data()