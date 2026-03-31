import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from safetensors.torch import load_file
import os
import mlflow

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
    
    # Отправляем готовую сетку как визуальный артефакт в MLflow
    mlflow.log_artifact(out_file)

# Список промптов из задания
prompts = [
    "<cheburashka> with the Eiffel Tower in the background",
    "<cheburashka> plushie",
    "<cheburashka> in sketch style",
    "<cheburashka> riding a bycycle"
]

model_id = "runwayml/stable-diffusion-v1-5"
# Попробуем сгенерировать результаты для промежуточного и финального чекпоинтов
checkpoints = [
    # "cheburashka_lora_checkpoint_100", 
    "cheburashka_lora_checkpoint_200", 
    # "cheburashka_lora_checkpoint_300", 
    "cheburashka_lora_checkpoint_400", 
    # "cheburashka_lora_checkpoint_500", 
    "cheburashka_lora_checkpoint_600", 
    # "cheburashka_lora_checkpoint_700", 
    "cheburashka_lora_checkpoint_800", 
    # "cheburashka_lora_checkpoint_900", 
    "cheburashka_lora_checkpoint_1000"
]

print("=== ЭТАП 3: Демонстрация результатов ===")

# Подключаемся к MLflow
#os.environ["NO_PROXY"] = "188.243.201.66,127.0.0.1,localhost"
mlflow.set_tracking_uri("http://188.243.201.66:5000")
mlflow.set_experiment("cheburashka-lora-inference-7")
mlflow.start_run()

for ckpt in checkpoints:
    print(f"\n==========================================")
    print(f"Загрузка пайплайна с подменённым UNet: {ckpt}")
    print(f"==========================================")
    
    # 1. Загружаем "чистый" UNet
    unet = UNet2DConditionModel.from_pretrained(
        model_id, 
        subfolder="unet", 
        torch_dtype=torch.float16
    ).to("cuda")
    
    # 2. Инициализируем ту же самую конфигурацию адаптера, что была на этапе 2
    # Это добавит в архитектуру UNet нужные слои, чтобы она в точности 
    # совпала с тем 1.8-гигабайтным "мутантом", который мы сохранили.
    lora_config = LoraConfig(
        r=16, # ДОЛЖЕН СОВПАДАТЬ С ПАРАМЕТРОМ r=16 ПРИ ОБУЧЕНИИ!
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )
    unet.add_adapter(lora_config)
    
    # 3. Напрямую загружаем веса в эту "подготовленную" архитектуру
    weights_path = os.path.join(ckpt, "lora_weights.pt")
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, weights_only=True)
        # Оставляем strict=False, так как стейт содержит только ключи lora_
        unet.load_state_dict(state_dict, strict=False)
    else:
        print(f"Файл весов не найден по пути: {weights_path}")
        continue
    
    # 4. Подключаем готовый кастомный UNet к пайплайну
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        unet=unet, 
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")
    
    # Запускаем генерацию
    generate_and_plot(pipe, prompts, ckpt)
    
    # Очищаем память перед загрузкой следующего чекпоинта
    del pipe
    del unet
    torch.cuda.empty_cache()

mlflow.end_run()
print("Готово! Все изображения сгенерированы и загружены в MLflow.")
