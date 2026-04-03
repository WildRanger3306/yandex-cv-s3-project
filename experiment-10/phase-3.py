import os
import gc

from diffusers import StableDiffusionPipeline

import torch

PATH_TO_IMAGES = 'data'
PATH_TO_ARTIFACTS = 'artifacts'

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Путь к сохранённой LoRA-модели из phase-2
final_save_path = "models/cheburashka_lora_final"

# Создадим снова пайплайн генерации и загрузим обученную модель.
# safety_checker=None — отключаем здесь, чтобы он не заменил изображение чёрным квадратом
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False
).to("cuda")

pipe.load_lora_weights(final_save_path)


prompt = "<cheburashka> plushie"
negative_prompt = "low quality, blurry"
with torch.no_grad():
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        height=512,
        width=512,
        guidance_scale=7.5
    ).images[0]
# Тестовая генерация — сохраняем чтобы проверить результат визуально
test_folder = os.path.join(PATH_TO_ARTIFACTS, 'test_inference')
os.makedirs(test_folder, exist_ok=True)
image.save(os.path.join(test_folder, 'cheburashka_test.png'))

del image
gc.collect()
torch.cuda.empty_cache() 
torch.cuda.synchronize()


prompts = [
    "<cheburashka> with the Eiffel Tower in the background",
    "<cheburashka> plushie",
    "<cheburashka> in sketch style",
    "<cheburashka> riding a bicycle"
]


negative_prompt = "low quality, blurry"

for i, prompt in enumerate(prompts):
    folder_to_save = os.path.join(PATH_TO_ARTIFACTS, 'after_train', f'prompt_{i}')
    os.makedirs(folder_to_save, exist_ok=True)
    for n in range(3):
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=30,
            height=512,   # SD 1.5 нативное разрешение — 1024 даёт тайлинг
            width=512,
            guidance_scale=7.5
        ).images[0]

        path_to_save = os.path.join(folder_to_save, f'cheburashka_{n}.png')
        image.save(path_to_save)

        del image
        gc.collect()
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()