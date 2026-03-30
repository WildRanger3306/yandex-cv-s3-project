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

def run_hires_generation(pipe_txt2img, pipe_img2img, prompt, neg_prompt):
    """
    Умная генерация: база 512 + мягкая доработка 768 со strength=0.35
    """
    with torch.autocast("cuda"):
        # 1. Базовая генерация ( identity )
        base_img = pipe_txt2img(
            prompt, 
            negative_prompt=neg_prompt,
            num_inference_steps=30, 
            guidance_scale=8.0
        ).images[0]
        
        # 2. Мягкая доработка деталей ( лицо/рот )
        upscaled = base_img.resize((768, 768), resample=Image.LANCZOS)
        refined = pipe_img2img(
            prompt=prompt,
            negative_prompt=neg_prompt,
            image=upscaled,
            strength=0.22, # Снижаем до ювелирного уровня
            guidance_scale=12.0, # Усиливаем влияние LoRA
            num_inference_steps=20
        ).images[0]
        
    return refined

# Список промптов из задания
prompts = [
    "<cheburashka> with the Eiffel Tower in the background",
    "<cheburashka> plushie",
    "<cheburashka> in sketch style",
    "<cheburashka> riding a bycycle"
]

model_id = "runwayml/stable-diffusion-v1-5"
# Попробуем сгенерировать результаты для промежуточного и финального чекпоинтов
# Список чекпоинтов для проверки (Experiment-8 генерировал каждые 200)
checkpoints = [
    "cheburashka_lora_checkpoint_200", 
    "cheburashka_lora_checkpoint_400", 
    "cheburashka_lora_checkpoint_600", 
    "cheburashka_lora_checkpoint_800", 
    "cheburashka_lora_checkpoint_1000"
]

print("=== ЭТАП 3: Демонстрация результатов ===")

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
    
    # 3. Собираем пайплайн с базовыми компонентами
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # 4. Нативная загрузка LoRA
    weights_path = os.path.join(ckpt, "lora_weights.pt")
    if os.path.exists(weights_path):
        print(f" -> Загрузка LoRA весов через Diffusers: {weights_path}")
        # Этот метод сам разберется с названиями слоев
        pipe.load_lora_weights(ckpt, weight_name="lora_weights.pt")
    else:
        print(f"Файл весов не найден: {weights_path}")
        continue
    
    # 5. Собираем Img2Img для рефайна, используя уже заряженный LoRA пайплайн
    from diffusers import StableDiffusionImg2ImgPipeline
    pipe_img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipe).to("cuda")

    # 6. Включаем Овердрайв (усиливаем LoRA, чтобы прогнать мишку)
    # scale=1.2 заставит Чебурашку проявиться сильнее
    cross_attention_kwargs = {"scale": 1.2}
    
    # Запускаем генерацию с использованием Hires Fix внутри цикла
    print(f"Генерация для: {ckpt}...")
    fig, axes = plt.subplots(1, len(prompts), figsize=(20, 5))
    out_dir = "results_final"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f" -> Промпт {i+1}: '{prompt}'")
        
        # Добавляем cross_attention_kwargs для управления силой LoRA
        with torch.autocast("cuda"):
            # 1. База
            base_img = pipe(
                prompt, negative_prompt=negative_prompt,
                num_inference_steps=30, guidance_scale=8.5,
                cross_attention_kwargs=cross_attention_kwargs
            ).images[0]
            
            # 2. Рефайн
            upscaled = base_img.resize((768, 768), resample=Image.LANCZOS)
            image = pipe_img2img(
                prompt=prompt, negative_prompt=negative_prompt,
                image=upscaled, strength=0.22, guidance_scale=12.0,
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
    grid_path = os.path.join(out_dir, f"FINAL_GRID_{ckpt}.png")
    plt.savefig(grid_path)
    mlflow.log_artifact(grid_path)
    plt.close()
    print(f"[!] Сетка сохранена: {grid_path}")
    
    # Очищаем память
    del pipe
    del pipe_img2img
    torch.cuda.empty_cache()

mlflow.end_run()
print("Готово! Все изображения сгенерированы и загружены в MLflow.")
