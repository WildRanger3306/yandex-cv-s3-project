import torch
import os
import mlflow
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
from peft import LoraConfig
from PIL import Image
import matplotlib.pyplot as plt

def generate_adetailer_style(pipe_txt2img, pipe_img2img, prompt, negative_prompt, ckpt_name):
    """
    Имитация Adetailer через двухэтапную генерацию (Hires. Fix)
    """
    print(f"[*] Генерация основы для: {ckpt_name}...")
    
    # 1. Первая фаза: Генерация 512x512
    with torch.autocast("cuda"):
        base_image = pipe_txt2img(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=30, 
            guidance_scale=8.5,
            width=512,
            height=512
        ).images[0]
    
    # 2. Апскейл до 768 (чтобы модели было больше места для деталей)
    upscaled_image = base_image.resize((768, 768), resample=Image.LANCZOS)
    
    # 3. Вторая фаза: Image-to-Image (Refining)
    # Усиливаем промпт для деталей лица
    refine_prompt = prompt + ", highly detailed face, visible smiling mouth, detailed character features"
    
    print(f"[*] Прорисовка деталей (Adetailer phase)...")
    with torch.autocast("cuda"):
        # denoising_strength=0.45 позволяет модели перерисовать "кашу" в детальные черты,
        # не меняя общую композицию тела и фона.
        refined_image = pipe_img2img(
            prompt=refine_prompt,
            negative_prompt=negative_prompt,
            image=upscaled_image,
            strength=0.45,  # Ключевой параметр: чем выше, тем больше изменений
            guidance_scale=8.5,
            num_inference_steps=20
        ).images[0]
    
    return base_image, refined_image

# Параметры
model_id = "runwayml/stable-diffusion-v1-5"
# Выбираем лучший чекпоинт по версии пользователя (600 или 800)
ckpt = "cheburashka_lora_checkpoint_800" 

# Промпты
prompts = [
    "<cheburashka> plushie, cute smile",
    "<cheburashka> riding a bicycle, outdoors",
    "<cheburashka> with the Eiffel Tower in the background, cinematic"
]

negative_prompt = (
    "bad anatomy, deformed face, missing mouth, blurred mouth, no mouth, "
    "empty face, two noses, extra ears, low quality, worst quality, blur, "
    "distorted face structure, scary eyes"
)

mlflow.set_tracking_uri("http://188.243.201.66:5000")
mlflow.set_experiment("cheburashka-lora-adetailer")

if __name__ == "__main__":
    print(f"=== ЭТАП 3: Adetailer Style (Refining Face) ===")
    
    # 1. Загрузка компонентов
    print(f"Загрузка весов из {ckpt}...")
    
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16).to("cuda")
    lora_config = LoraConfig(r=16, target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet.add_adapter(lora_config)
    
    weights_path = os.path.join(ckpt, "lora_weights.pt")
    if os.path.exists(weights_path):
        unet.load_state_dict(torch.load(weights_path, weights_only=True), strict=False)
        
    text_encoder_path = os.path.join(ckpt, "text_encoder")
    if os.path.exists(text_encoder_path):
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16).to("cuda")
    else:
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=torch.float16).to("cuda")

    # 2. Инициализация пайплайнов
    pipe_txt2img = StableDiffusionPipeline.from_pretrained(
        model_id, unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    
    # Используем те же компоненты для Img2Img, чтобы не дублировать память
    pipe_img2img = StableDiffusionImg2ImgPipeline(
        vae=pipe_txt2img.vae,
        text_encoder=pipe_txt2img.text_encoder,
        tokenizer=pipe_txt2img.tokenizer,
        unet=pipe_txt2img.unet,
        scheduler=pipe_txt2img.scheduler,
        feature_extractor=pipe_txt2img.feature_extractor,
        safety_checker=None
    ).to("cuda")

    with mlflow.start_run():
        os.makedirs("results_adetailer", exist_ok=True)
        
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing Prompt {i+1}: {prompt}")
            base, refined = generate_adetailer_style(pipe_txt2img, pipe_img2img, prompt, negative_prompt, ckpt)
            
            # Сохранение и сравнение
            comparison = Image.new('RGB', (1024 + 10, 512)) # Сетка для сравнения
            comparison.paste(base, (0, 0))
            # Для сравнения уменьшим refined обратно до 512 (или отобразим фрагмент)
            comparison.paste(refined.resize((512, 512)), (522, 0))
            
            path = f"results_adetailer/comparison_{i}.png"
            comparison.save(path)
            mlflow.log_artifact(path)
            
            # Одиночные файлы
            refined.save(f"results_adetailer/refined_{i}.png")
            mlflow.log_artifact(f"results_adetailer/refined_{i}.png")

    print("\nГотово! Результаты в папке 'results_adetailer' и в MLflow.")
