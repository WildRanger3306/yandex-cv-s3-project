"""
Скрипт для генерации индивидуальных текстовых описаний (captions) к каждой картинке датасета.
Запускать ОДИН РАЗ перед началом обучения.

Принцип: к каждому файлу cheburashka_N.png создается cheburashka_N.txt с уникальным описанием.
Формат: "<cheburashka> plushie [описание фона/контекста]"

Уникальное слово <cheburashka> привязывается к персонажу,
а всё остальное описание помогает модели понять фон и контекст.
"""

import os

# Папка с датасетом (относительно корня проекта, запускать из корня)
DATA_DIR = "../dataset-2"

# Базовый описатель — уникальный токен + класс (стандарт DreamBooth)
BASE_TOKEN = "<cheburashka> plushie"

# Вручную написанные описания для каждой картинки.
# Ключ — имя файла, значение — полное описание.
# Если картинки нет в словаре — используется BASE_TOKEN как фоллбэк.
CAPTIONS = {
    "cheburashka_1.png":  f"{BASE_TOKEN}, movie poster, blue background, group of people",
    "cheburashka_2.png":  f"{BASE_TOKEN}, Look surprised, against the background of an old house",
    "cheburashka_3.png":  f"{BASE_TOKEN}, studio shot, in the children's room, front view",
    "cheburashka_4.png":  f"{BASE_TOKEN}, outdoor scene, natural light",
    "cheburashka_5.png":  f"{BASE_TOKEN}, Sitting on the board, holding a tangerine",
    "cheburashka_6.png":  f"{BASE_TOKEN}, in the garden, standing on a basket of tangerines",
    # "cheburashka_7.png":  f"{BASE_TOKEN}, standing pose, full body",
    "cheburashka_8.png":  f"{BASE_TOKEN}, studio shot, gym, furry, standing tall, holding a basketball",
    "cheburashka_9.png":  f"{BASE_TOKEN}, greets,against the background of wooden chairs",
    "cheburashka_10.png": f"{BASE_TOKEN}, movie poster, with a winter landscape in the background, holding a tangerine",
    "cheburashka_11.png": f"{BASE_TOKEN}, In the garden, soft light",
    "cheburashka_12.png": f"{BASE_TOKEN}, sliding down a rubber slide",
    "cheburashka_13.png": f"{BASE_TOKEN}, against the background of green grass and tangerine peels",
    "cheburashka_14.png": f"{BASE_TOKEN}, he has a glass of orange juice in his room",
    "cheburashka_15.png": f"{BASE_TOKEN}, in the garden, portrait shot, soft light",
    "cheburashka_16.png": f"{BASE_TOKEN}, in the garden, portrait shot, soft light",
    "cheburashka_17.png": f"{BASE_TOKEN}, in a wooden box on a motorcycle",
    "cheburashka_18.png": f"{BASE_TOKEN}, he squinted in the room, holding a tangerine",
    "cheburashka_19.png": f"{BASE_TOKEN}, standing on the sidewalk, looking up",
    "cheburashka_20.png": f"{BASE_TOKEN}, He holds a soccer ball in the park.",
    "cheburashka_21.png": f"{BASE_TOKEN}, He holds a soccer ball, on the sidewalk",
    "cheburashka_22.png": f"{BASE_TOKEN}, standing on the sidewalk, looking up",
    # "cheburashka_23.png": f"{BASE_TOKEN}, close up, detailed fur",
    "cheburashka_24.png": f"{BASE_TOKEN}, action pose, dynamic",
    "cheburashka_25.png": f"{BASE_TOKEN}, indoor scene, soft lighting, furry",
    "cheburashka_26.png": f"{BASE_TOKEN}, indoor scene, soft lighting, furry",
    # "cheburashka_27.png": f"{BASE_TOKEN}, holding object, interactive pose",
    "cheburashka_28.png": f"{BASE_TOKEN}, group scene, background characters",
}

def generate_captions(data_dir):
    if not os.path.exists(data_dir):
        print(f"Ошибка: папка '{data_dir}' не найдена!")
        return

    images = sorted([f for f in os.listdir(data_dir) if f.lower().endswith('.png')])
    
    if not images:
        print(f"В папке '{data_dir}' не найдено PNG-файлов!")
        return

    print(f"Найдено {len(images)} изображений. Генерация captions...")
    print("-" * 50)

    for img_file in images:
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(data_dir, txt_file)

        # Берем описание из словаря, или используем базовый токен
        caption = CAPTIONS.get(img_file, BASE_TOKEN)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(caption)

        print(f"  {img_file} → {txt_file}: \"{caption}\"")

    print("-" * 50)
    print(f"✅ Готово! Создано {len(images)} файлов описаний в '{data_dir}'")
    print("Теперь можно запускать phase-2.py для обучения с captioning!")

if __name__ == "__main__":
    generate_captions(DATA_DIR)
