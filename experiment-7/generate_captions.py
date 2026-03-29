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
    "cheburashka_1.png":  f"{BASE_TOKEN}, white background, studio shot",
    "cheburashka_2.png":  f"{BASE_TOKEN}, white background, studio shot, side view",
    "cheburashka_3.png":  f"{BASE_TOKEN}, white background, studio shot, front view",
    "cheburashka_4.png":  f"{BASE_TOKEN}, outdoor scene, natural light",
    "cheburashka_5.png":  f"{BASE_TOKEN}, close up face portrait",
    "cheburashka_6.png":  f"{BASE_TOKEN}, sitting pose, colorful background",
    "cheburashka_7.png":  f"{BASE_TOKEN}, standing pose, full body",
    "cheburashka_8.png":  f"{BASE_TOKEN}, outdoor scene, green background",
    "cheburashka_9.png":  f"{BASE_TOKEN}, dramatic lighting, cinematic",
    "cheburashka_10.png": f"{BASE_TOKEN}, movie scene, dark background",
    "cheburashka_11.png": f"{BASE_TOKEN}, movie poster, jungle background",
    "cheburashka_12.png": f"{BASE_TOKEN}, basketball court, sports arena",
    "cheburashka_13.png": f"{BASE_TOKEN}, close up, big eyes, furry",
    "cheburashka_14.png": f"{BASE_TOKEN}, adventure scene, outdoor",
    "cheburashka_15.png": f"{BASE_TOKEN}, smiling, happy expression",
    "cheburashka_16.png": f"{BASE_TOKEN}, wooden background, warm lighting",
    "cheburashka_17.png": f"{BASE_TOKEN}, full body shot, standing",
    "cheburashka_18.png": f"{BASE_TOKEN}, cute pose, big ears visible",
    "cheburashka_19.png": f"{BASE_TOKEN}, movie still, dramatic scene",
    "cheburashka_20.png": f"{BASE_TOKEN}, colorful background, vivid colors",
    "cheburashka_21.png": f"{BASE_TOKEN}, side profile, fur texture",
    "cheburashka_22.png": f"{BASE_TOKEN}, looking up, sky background",
    "cheburashka_23.png": f"{BASE_TOKEN}, close up, detailed fur",
    "cheburashka_24.png": f"{BASE_TOKEN}, action pose, dynamic",
    "cheburashka_25.png": f"{BASE_TOKEN}, indoor scene, soft lighting",
    "cheburashka_26.png": f"{BASE_TOKEN}, movie promotional art",
    "cheburashka_27.png": f"{BASE_TOKEN}, holding object, interactive pose",
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
