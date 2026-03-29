import os
import requests
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from diffusers import StableDiffusionPipeline

def download_dataset():
    """Скачивание датасета с чебурашкой"""
    os.makedirs("dataset", exist_ok=True)
    urls = [
        "https://code.s3.yandex.net/deep-learning/cheburashka_1.png?etag=c5d0692223b9a83575ef4ccee9679c54",
        "https://code.s3.yandex.net/deep-learning/cheburashka_2.png?etag=dc3c5bb94cd1571c0ce7cc5a540500ab",
        "https://code.s3.yandex.net/deep-learning/cheburashka_3.png?etag=3bd86d39bacaf6d71daa580b676212fa"
    ]
    image_paths = []
    for i, url in enumerate(urls):
        path = f"dataset/cheburashka_{i+1}.png"
        if not os.path.exists(path):
            print(f"Скачивание {path}...")
            resp = requests.get(url)
            with open(path, "wb") as f:
                f.write(resp.content)
        image_paths.append(path)
    return image_paths

def load_dataset_2(data_dir="dataset-2"):
    """Загрузка путей к картинкам из локальной папки dataset-2"""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []
    paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return sorted(paths)

def visualize_dataset(image_paths):
    display_paths = image_paths[:15]
    if not display_paths:
        return
        
    # Сетка: 3 строки на 5 колонок
    fig, axes = plt.subplots(3, 5, figsize=(20, 12))
    axes = axes.flatten() # Делаем одномерным для прохода в цикле
        
    for idx, path in enumerate(display_paths):
        img = Image.open(path).convert("RGB")
        axes[idx].imshow(img)
        axes[idx].axis("off")
        axes[idx].set_title(os.path.basename(path)[:20]) # обрезаем слишком длинные имена
        
    # Скрываем пустые слоты (на случай если картинок < 15)
    for idx in range(len(display_paths), len(axes)):
        axes[idx].axis("off")
        
    plt.tight_layout()
    plt.savefig("dataset_visualization.png")
    print(f"Визуализация (сетка 3x5, {len(display_paths)} картинок) сохранена в 'dataset_visualization.png'\n")

# 1. Работа с данными
print("=== ЭТАП 1: Работа с данными ===")
image_paths = load_dataset_2()
# if not image_paths:
#     print("Папка dataset_2 пуста, качаем дефолтный датасет...")
#     image_paths = download_dataset()
visualize_dataset(image_paths)

# Реализация класса датасета
class CheburashkaDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = image_paths
        # Ресайз к 512x512, преобразование в тензор и нормализация (-1 до 1) 
        # Mean 0.5 и Std 0.5 переводит [0, 1] в [-1, 1] -- стандарт для SD1.5
        self.transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img)

dataset = CheburashkaDataset(image_paths)
print(f"Создан датасет размером: {len(dataset)}")
sample_tensor = dataset[0]
print(f"Форма тензора первого элемента: {sample_tensor.shape}\n")


# 2. Работа с оригинальной моделью
print("=== ЭТАП 2: Работа с моделью ===")
print("Загрузка модели Stable Diffusion 1.5...")
# Используем fp16. Для 1080Ti на 11 ГБ этого более чем достаточно,
# SD 1.5 весит ~3-4 ГБ в видеопамяти при dtype=float16.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# Генерация с сырой моделью
prompt = "<cheburashka> with the Eiffel Tower in the background"
print(f"Генерация тестового изображения по промпту: '{prompt}'...")
# Генерируем всего одну картинку
raw_image = pipe(prompt=prompt, num_inference_steps=30).images[0]
raw_image.save("raw_model_generation.png")
print("Изображение было сохранено в 'raw_model_generation.png' (Чебурашки там быть не должно)\n")

# Извлекаем текстовые эмбеддинги
print("Исследование функции encode_prompt и извлечение эмбеддингов...")
target_prompt = "<cheburashka> plushie"
target_device = pipe.device

# Вызываем encode_prompt. 
# Возвращает кортеж (prompt_embeds, negative_prompt_embeds)
prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
    prompt=target_prompt,
    device=target_device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=False
)

torch.save(prompt_embeds, "cheburashka_embeds.pt")
print(f"Текстовые эмбеддинги для промпта '{target_prompt}' сохранены в 'cheburashka_embeds.pt'")
print(f"Форма сохраненного тензора: {prompt_embeds.shape}")

print("\n--- Phase 1 завершена успешно ---")
