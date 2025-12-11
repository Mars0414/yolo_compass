import os
import shutil
import random
from pathlib import Path

# --- НАСТРОЙКИ ---
# Папка с картинками
SOURCE_IMAGES_DIR = 'frames_v5'

# Папка с разметкой (txt)
SOURCE_LABELS_DIR = 'labes_v5'

# Куда класть (папка основного датасета)
DATASET_DIR = 'dataset'

# Сколько процентов отдать на проверку (0.2 = 20%)
VAL_RATIO = 0.2

# Префикс, чтобы файлы не перезаписались (меняй при добавлении новых видео!)
# Например: 'video2_', 'test_video_'
FILE_PREFIX = 'v5_'


# -----------------

def split_and_merge_data():
    src_img_path = Path(SOURCE_IMAGES_DIR)
    src_lbl_path = Path(SOURCE_LABELS_DIR)
    dataset_path = Path(DATASET_DIR)

    # Проверки существования папок
    if not src_img_path.exists():
        print(f"Ошибка: Папка с картинками {SOURCE_IMAGES_DIR} не найдена!")
        return
    if not src_lbl_path.exists():
        print(f"Ошибка: Папка с разметкой {SOURCE_LABELS_DIR} не найдена!")
        return

    # Собираем пары
    pairs = []  # Список кортежей (путь_к_jpg, путь_к_txt)

    # Ищем все jpg в папке картинок
    jpg_files = list(src_img_path.glob('*.jpg'))

    print(f"Всего картинок в {SOURCE_IMAGES_DIR}: {len(jpg_files)}")

    for jpg_file in jpg_files:
        # Ищем файл с таким же именем, но .txt в папке разметки
        txt_name = f"{jpg_file.stem}.txt"
        txt_file = src_lbl_path / txt_name

        if txt_file.exists():
            pairs.append((jpg_file, txt_file))
        else:
            # Если картинка есть, а разметки нет — пропускаем
            pass

    if not pairs:
        print(f"Не найдено совпадений! Проверь имена файлов.")
        print(f"Пример: если есть '{SOURCE_IMAGES_DIR}/frame_0.jpg',")
        print(f"то должен быть '{SOURCE_LABELS_DIR}/frame_0.txt'")
        return

    # Перемешиваем
    random.shuffle(pairs)

    # Делим на train и valid
    num_val = int(len(pairs) * VAL_RATIO)
    val_pairs = pairs[:num_val]
    train_pairs = pairs[num_val:]

    print(f"Найдено готовых пар (фото+txt): {len(pairs)}")
    print(f"В обучение (train): {len(train_pairs)}")
    print(f"В проверку (valid): {len(val_pairs)}")

    # Функция перемещения
    def move_files(pair_list, split_type):
        dest_images = dataset_path / split_type / 'images'
        dest_labels = dataset_path / split_type / 'labels'

        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        count = 0
        for jpg_path, txt_path in pair_list:
            # Новые имена
            new_jpg_name = f"{FILE_PREFIX}{jpg_path.name}"
            new_txt_name = f"{FILE_PREFIX}{txt_path.name}"

            # Перемещаем (shutil.move переносит файл, shutil.copy копирует)
            # Лучше использовать copy, чтобы исходники остались на всякий случай
            shutil.copy(str(jpg_path), str(dest_images / new_jpg_name))
            shutil.copy(str(txt_path), str(dest_labels / new_txt_name))
            count += 1
        return count

    print("Копирую файлы...")
    # Проверь, как называется папка валидации в dataset (val или valid?)
    # Обычно в yolo стурктуре это 'valid' или 'val'
    t_count = move_files(train_pairs, 'train')
    v_count = move_files(val_pairs, 'valid')

    print(f"Готово! Скопировано {t_count + v_count} пар файлов.")
    print("Исходные файлы остались на месте (я использовал копирование).")


if __name__ == "__main__":
    split_and_merge_data()