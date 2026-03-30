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
# Полный список чекпоинтов для анализа динамики
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

    # 3. Загружаем дообученный Text Encoder (РАЗУМ МОДЕЛИ)
    text_encoder_path = os.path.join(ckpt, "text_encoder")
    if os.path.exists(text_encoder_path):
        print(f" -> Загрузка дообученного Text Encoder из {text_encoder_path}")
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16).to("cuda")
    else:
        print(" -> ВНИМАНИЕ: Дообученный Text Encoder не найден, использую базовый")
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")

    # 4. Загрузка весов LoRA напрямую в unet
    weights_path = os.path.join(ckpt, "lora_weights.pt")
    if os.path.exists(weights_path):
        print(f" -> Точная загрузка весов в UNet: {weights_path}")
        state_dict = torch.load(weights_path, weights_only=True)
        try:
            # Сначала пробуем строгую загрузку, чтобы поймать несовпадение имен
            unet.load_state_dict(state_dict, strict=True)
            print(" -> [OK] Веса успешно загружены (strict=True)")
        except Exception as e:
            print(f" -> [!] Ошибка загрузки: {str(e)[:500]}...")
            print(" -> Попытка мягкой загрузки (strict=False)...")
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

    # scale=1.1 (чуть выше нормы для уверенного образа)
    cross_attention_kwargs = {"scale": 1.1}
    
    # Запускаем генерацию с использованием Hires Fix внутри цикла
    print(f"Генерация для: {ckpt}...")
    fig, axes = plt.subplots(1, len(prompts), figsize=(20, 5))
    out_dir = "results_refined_v2"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f" -> Промпт {i+1}: '{prompt}'")
        
        with torch.autocast("cuda"):
            # 1. База ( identity )
            base_img = pipe(
                prompt, negative_prompt=negative_prompt,
                num_inference_steps=30, guidance_scale=8.0,
                cross_attention_kwargs=cross_attention_kwargs
            ).images[0]
            
            # 2. Рефайн ( микро-доработка рта )
            upscaled = base_img.resize((768, 768), resample=Image.LANCZOS)
            image = pipe_img2img(
                prompt=prompt, negative_prompt=negative_prompt,
                image=upscaled, 
                strength=0.18, # Минимум вмешательства
                guidance_scale=8.0, # Натуральный контраст
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
    del unet
    del text_encoder
    torch.cuda.empty_cache()

mlflow.end_run()
print("Готово! Все изображения сгенерированы и загружены в MLflow.")
