import os
import gc

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, PeftModel, get_peft_model_state_dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from utilits import Visualisator, CheburashkaDataset, save_checkpoint, draw_loss_graph
import numpy as np

PATH_TO_IMAGES = 'data'
PATH_TO_ARTIFACTS = 'artifacts'

torch.manual_seed(42)
np.random.seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


# Загрузим модель и настроим LoRA
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to(device)

lora_config = LoraConfig(
    r=128, 
    target_modules=["to_k", "to_q", "to_v", "to_out.0"], 
    init_lora_weights="gaussian",
    lora_dropout=0.1
)

# Объединение LoRa и основной модели. Заморозка весов. Перевод в режим тренировки.
pipe.unet.add_adapter(lora_config)

pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)

for param in pipe.unet.parameters():
    param.requires_grad = False

for name, param in pipe.unet.named_parameters():
    if "lora" in name.lower():
        param.requires_grad = True

pipe.unet.train()


# Создадим датасет и даталоадер.
files = os.listdir(PATH_TO_IMAGES)
image_paths = [
    os.path.join(PATH_TO_IMAGES, img) 
    for img in files 
    if os.path.splitext(img)[1] in ['.png', '.jpg']
]

dataset = CheburashkaDataset(image_paths, size=512)
train_dataloader = DataLoader(
    dataset, 
    batch_size=1,
    shuffle=True
)


# Загрузим полученные ранее эмбединги
embeddings_data = torch.load("embeddings/cheburashka_plushie_embeddings.pt")
prompt_embeds = embeddings_data["prompt_embeds"].to("cuda", dtype=torch.float32)
negative_prompt_embeds = embeddings_data["negative_prompt_embeds"].to("cuda", dtype=torch.float32)

# Настроим оптимизатор, планировщик скорости обучения и планировщик диффузии.
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, pipe.unet.parameters()),
    lr=2.0e-05,
    weight_decay=1e-2,
    eps=1e-08
)

lr_scheduler = get_scheduler(
    "constant",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=1000
)

noise_scheduler = DDPMScheduler.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    subfolder="scheduler"
)

scaler = torch.cuda.amp.GradScaler()

# Помещаем VAE и текстовый энкодор на ГПУ, переводим их в режим оценки, т.к. они не учатся.

pipe.vae = pipe.vae.to("cuda")
pipe.text_encoder = pipe.text_encoder.to("cuda")
pipe.vae.eval()
pipe.text_encoder.eval()

# Задаём параметры обучения и создаём прогремм бар для визуализации
max_train_steps = 1000
snr_gamma = 5.0
losses = []


progress_bar = tqdm(range(max_train_steps), desc="Обучение")
global_step = 0

# Задаём параметры обучения и создаём прогремм бар для визуализации
while global_step < max_train_steps:
    for batch in train_dataloader:
        # VAE 
        # Кодирует изображения в латентное пространство меньшей размерности.
        # Это ускоряет обучение и уменьшает потребление памяти.
        with torch.no_grad():
            latents = pipe.vae.encode(batch.to("cuda")).latent_dist.sample()
            latents = latents * pipe.vae.config.scaling_factor
        
        # Семплирование диффузионного времени
        # Случайно выбираем время t из равномерного распределения [0, 1), 
        # для опредления уровеня зашумленности (0 - чистое изображение, 1 - чистый шум)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, 
            (latents.shape[0],), device=latents.device
        ).long()
        
        # Зашумление изображения
        noise = torch.randn_like(latents)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # Подготовка входных данных для основной модели (UNet)
        # На вход UNet подаются:
        # - зашумленные латенты
        # - время диффузии
        # - текстовые эмбеддинги
        batch_size = latents.shape[0]
        encoder_hidden_states = prompt_embeds.repeat(batch_size, 1, 1)

        # Прямой проход
        # Модель предсказывает шум, который был добавлен ранее
        noise_pred = pipe.unet(
            noisy_latents, timesteps, encoder_hidden_states
        ).sample

        # Расчет функции потерь     
        if snr_gamma is not None:
            # Вычисляем SNR 
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(latents.device)
            snr = alphas_cumprod[timesteps] / (1 - alphas_cumprod[timesteps])
            # Вычисляем веса для каждого примера
            snr_weight = torch.where(snr < snr_gamma, snr, torch.ones_like(snr) * snr_gamma)
            snr_weight = snr_weight / snr_gamma
            # Взвешенный MSE loss
            loss = F.mse_loss(noise_pred, noise, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weight
            loss = loss.mean()
        else:
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
                
        # Обратное распространение
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1)
        
        optimizer.step()
        optimizer.zero_grad()
        
        losses.append(loss.detach().item())
        progress_bar.update(1)
        progress_bar.set_postfix({"loss": losses[-1], "step": global_step})
        
        # Сохраняем чекпоинты
        if (global_step + 1) % 500 == 0:
            save_checkpoint(
                pipe, 
                optimizer, 
                global_step + 1
            )
        
        global_step += 1
        
        # выходим из ципкла, если достигнутое заданое количество шагов
        if global_step >= max_train_steps:
            break

progress_bar.close()

# Сохраним финальную модель с метаинформацией.
final_save_path = "models/cheburashka_lora_final"
os.makedirs(final_save_path, exist_ok=True)

lora_state_dict = get_peft_model_state_dict(pipe.unet)
pipe.save_lora_weights(
    save_directory=final_save_path,
    unet_lora_layers=lora_state_dict,
    safe_serialization=True # Сохранит в современном формате .safetensors
)

config = {
    "lora_rank": 128,
    "trained_steps": max_train_steps,
    "learning_rate": 2.0e-05,
}
torch.save(config, os.path.join(final_save_path, "training_config.pt"))


draw_loss_graph(losses, os.path.join(PATH_TO_ARTIFACTS, 'loss.png'))

# Высвобождаем ресурсы
del pipe 
del optimizer, lr_scheduler, noise_scheduler
del batch, latents, noisy_latents, noise, noise_pred, encoder_hidden_states, timesteps
del train_dataloader
del progress_bar

gc.collect()
torch.cuda.empty_cache() 
torch.cuda.synchronize()