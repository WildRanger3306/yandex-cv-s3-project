import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from safetensors.torch import load_file
import os
import mlflow
from transformers import CLIPTextModel
from PIL import Image

def generate_and_plot(pipe, prompts, checkpoint_name):
    """
    Генерирует изображения по списку промптов и сохраняет их в один общий файл
    """
    print(f"Генерация для: {checkpoint_name}...")
    fig, axes = plt.subplots(1, len(prompts), figsize=(20, 5))
    
    out_dir = "results_local"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f" -> Промпт {i+1}: '{prompt}'")
        # Генерируем картинку. guidance_scale=7.5 — стандарт для Stable Diffusion
        image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        
        # Отрисовка
        axes[i].imshow(image)
        axes[i].set_title(prompt, fontsize=9, wrap=True)
        axes[i].axis("off")
        
        # Дополнительно сохраним каждую картинку отдельно локально
        safe_prompt = prompt.replace(" ", "_").replace("<", "").replace(">", "").replace(":", "")
        local_img_path = os.path.join(out_dir, f"{checkpoint_name}_{safe_prompt[:25]}.png")
        image.save(local_img_path)
        
        # Логируем заодно и индивидуальные картинки в MLflow
        mlflow.log_artifact(local_img_path)
        
    plt.tight_layout()
    out_file = os.path.join(out_dir, f"GRID_{checkpoint_name}.png")
    plt.savefig(out_file)
    print(f"[*] Склейка сохранена в файл: {out_file}\n")
    plt.close()
    
    mlflow.log_artifact(out_file)

# Список промптов из задания (Proxy Image Generation)
prompts = [
    "<cheburashka> with the Eiffel Tower in the background",
    "<cheburashka> plushie, stuffed toy, made of fabric, button eyes, lifeless, stitching, toy on shelf", 
    "<cheburashka> in sketch style",
    "<cheburashka> riding a bicycle, full body, cycling in the park, holding handlebars"
]

model_id = "runwayml/stable-diffusion-v1-5"
# Попробуем сгенерировать результаты для промежуточного и финального чекпоинтов
checkpoints = [
    "cheburashka_lora_checkpoint_200", 
    "cheburashka_lora_checkpoint_400", 
    "cheburashka_lora_checkpoint_600", 
    "cheburashka_lora_checkpoint_800", 
    "cheburashka_lora_checkpoint_1000"
]

print("=== ЭТАП 3: Proxy-генерация ===")

# Негативные промпты для спасения лица
negative_prompt = (
    "bad anatomy, deformed face, missing mouth, blurred mouth, no mouth, "
    "two noses, extra ears, low quality, worst quality, blur, distortion, "
    "scary face, messy face"
)

# Подключаемся к MLflow
mlflow.set_tracking_uri("http://188.243.201.66:5000")
mlflow.set_experiment("cheburashka-lora-final-results")
mlflow.start_run()

for ckpt in checkpoints:
    print(f"\n==========================================")
    print(f"Загрузка пайплайна с подменённым UNet: {ckpt}")
    print(f"==========================================")
    
    # 1. Загружаем чистый UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id, subfolder="unet", torch_dtype=torch.float16
    ).to("cuda")
    
    # 2. Добавляем тот же адаптер, что был при обучении
    lora_config = LoraConfig(
        r=16, 
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(lora_config)

    # 3. ВСЕГДА используем базовый кодировщик (для чистоты теста)
    print(f" -> Использую БАЗОВЫЙ Text Encoder v1.5")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")

    # 4. Загрузка LoRA весов
    weights_path = os.path.join(ckpt, "lora_weights.pt")
    if os.path.exists(weights_path):
        print(f" -> Загрузка весов: {weights_path}")
        state_dict = torch.load(weights_path, weights_only=True)
        unet.load_state_dict(state_dict, strict=False)
    else:
        print(f"Файл весов не найден: {weights_path}")
        continue
    
    # 5. Собираем пайплайн
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        unet=unet,
        text_encoder=text_encoder,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # 5. Собираем Img2Img для рефайна
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipe).to("cuda")

    print(f"Генерация (LORA + PROXY REFINER) для: {ckpt}...")
    fig, axes = plt.subplots(1, len(prompts), figsize=(20, 5))
    out_dir = "results_lora_proxy"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f" -> Промпт {i+1}: '{prompt}'")
        
        lower_prompt = prompt.lower()
        is_sketch = "sketch" in lower_prompt
        is_plushie = "plushie" in lower_prompt or "toy" in lower_prompt
        is_bicycle = "bicycle" in lower_prompt
        
        # Настройка базовых весов
        if is_sketch:
            lora_scale = 0.75
        elif is_plushie:
            lora_scale = 1.3 # Умеренная сила для игрушки
        elif is_bicycle:
            lora_scale = 1.5 # LoRA включится только на ВТОРОМ этапе (Img2Img)
        else:
            lora_scale = 1.6 # Максимальная сила для живых персонажей
            
        cross_attention_kwargs = {"scale": lora_scale}
        
        # Настройка негативных промптов
        current_neg_prompt = negative_prompt
        if is_sketch:
            current_neg_prompt += ", photo, real background, rocks, stones, plants, 3d render, realistic textures"
        
        if is_plushie:
            current_neg_prompt += ", alive, organic, real animal, eye reflection, wet nose, emotional eyes, movie character"
            
        with torch.autocast("cuda"):
            if is_bicycle:
                # ВЕЛОСИПЕД: Хитрый двухэтапный процесс (Proxy Generation)
                print("    [*] Шаг 2 (Полировка): Запуск Proxy Generation для велосипеда...")
                
                # 1. Генерируем "болванку" (Silhouette Match)
                # Просим огромные уши СРАЗУ, чтобы базовый велик их учитывал
                proxy_prompt = "a small fluffy creature with huge oversized round ears riding a bicycle, full body, cycling in the park, hands on handlebars, high quality"
                base_img = pipe(
                    proxy_prompt, negative_prompt=current_neg_prompt,
                    num_inference_steps=35, guidance_scale=7.5,
                    cross_attention_kwargs={"scale": 0.0} # LoRA ОТКЛЮЧЕНА
                ).images[0]
                
                # 2. Превращаем болванку в Чебурашку (Сильная LoRA + Img2Img)
                upscaled = base_img.resize((768, 768), resample=Image.LANCZOS)
                image = pipe_img2img(
                    prompt=prompt, negative_prompt=current_neg_prompt,
                    image=upscaled, 
                    strength=0.72, # Повышаем силу трансформации (чтобы 5/10 превратилось в 10/10)
                    guidance_scale=12.0, # Усиливаем пробитие промпта
                    cross_attention_kwargs={"scale": 1.7} # Овердрайв Лоры для велосипеда
                ).images[0]
                
            else:
                # 1. Базовая генерация для Чебурашки (с LoRA)
                base_img = pipe(
                    prompt, negative_prompt=current_neg_prompt,
                    num_inference_steps=30, guidance_scale=8.5,
                    cross_attention_kwargs=cross_attention_kwargs
                ).images[0]
                
                if is_sketch:
                    # Для скетча НЕ делаем рефайн
                    image = base_img 
                else:
                    # 2. Hires. Fix (Ремонт деталей лица и рта для всех кроме велосипеда и скетча)
                    upscaled = base_img.resize((768, 768), resample=Image.LANCZOS)
                    image = pipe_img2img(
                        prompt=prompt, negative_prompt=current_neg_prompt,
                        image=upscaled, 
                        strength=0.22, # Ювелирная точность для лица
                        guidance_scale=8.5,
                        cross_attention_kwargs=cross_attention_kwargs
                    ).images[0]
        
        axes[i].imshow(image)
        axes[i].set_title(prompt, fontsize=9, wrap=True)
        axes[i].axis("off")
        
        safe_prompt = prompt.replace(" ", "_").replace("<", "").replace(">", "").strip("_")[:20]
        local_path = os.path.join(out_dir, f"{ckpt}_{safe_prompt}.png")
        image.save(local_path)
        mlflow.log_artifact(local_path)
        
    plt.tight_layout()
    grid_path = os.path.join(out_dir, f"FINAL_GRID_PROXY_{ckpt}.png")
    plt.savefig(grid_path)
    mlflow.log_artifact(grid_path)
    plt.close()
    print(f"[!] Сетка сохранена: {grid_path}")
    
    # Очистка
    del pipe
    del pipe_img2img
    del unet
    torch.cuda.empty_cache()

mlflow.end_run()
print("Готово! Все изображения сгенерированы и загружены в MLflow.")
