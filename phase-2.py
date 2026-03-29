import os
import torch
import torch.nn.functional as F
import mlflow
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from peft import LoraConfig

# ================================
# 1. Данные и Датасет
# ================================
class CheburashkaDataset(Dataset):
    def __init__(self, data_dir="dataset"):
        # Собираем изображения из папки dataset (подготовлено на этапе 1)
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.png')]
        self.transform = T.Compose([
            T.Resize(512),         # Сжимаем изображение (сохраняя пропорции!), чтобы меньшая сторона была 512
            T.CenterCrop(512),     # Отсекаем "лишнее" по бокам, строго выделяя центр квадратом 512x512
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)

print("Загрузка датасета...")
dataset = CheburashkaDataset("dataset-2")
train_batch_size = 1
dataloader = DataLoader(dataset, batch_size=train_batch_size, shuffle=True)

# Загружаем эмбеддинги условия (из этапа 1)
print("Чтение текстовых эмбеддингов 'cheburashka plushie'...")
prompt_embeds = torch.load("cheburashka_embeds.pt").to("cuda")

# ================================
# 2. Загрузка моделей
# ================================
print("Загрузка базовой модели SD 1.5...")
model_id = "runwayml/stable-diffusion-v1-5"
weight_dtype = torch.float16 # fp16 для экономии VRAM (11 ГБ вполне достаточно)

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype).to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=weight_dtype).to("cuda")

# Зафиксируем веса VAE и UNet: мы не обучаем их "целиком"
vae.requires_grad_(False)
unet.requires_grad_(False)

# ================================
# 3. Добавление LoRA
# ================================
print("Инициализация адаптеров LoRA (PEFT)...")
lora_rank = 128
lora_config = LoraConfig(
    r=lora_rank,
    init_lora_weights="gaussian",
    # Модули внимания, на которые мы вешаем проекции
    target_modules=["to_k", "to_q", "to_v", "to_out.0"] 
)

# Инжект LoRA адаптера в базовый UNet
unet.add_adapter(lora_config)

# Убедимся, что новые обучаемые веса LoRA созданы в fp32 (для устойчивости)
for param in unet.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

# ================================
# 4. Настройка обучения
# ================================
# Для LoRA стандартный LR значительно выше (обычно 1e-4 - 2e-4) чем для файн-тюнинга Unet
learning_rate = 1.5e-4 
# Удвоили количество шагов для лучшего закрепления черт объекта
max_train_steps = 2000 
max_grad_norm = 1.0

# В оптимизатор передаются только слои с requires_grad=True (LoRA слои)
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, unet.parameters()),
    lr=learning_rate
)

unet.train()
global_step = 0
progress_bar = tqdm(total=max_train_steps, desc="Обучение")

# ================================
# Инициализация MLflow для логирования
# ================================
os.environ["NO_PROXY"] = "188.243.201.66,127.0.0.1,localhost"
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("cheburashka-lora-training")
mlflow.start_run()
mlflow.log_params({
    "learning_rate": learning_rate,
    "max_train_steps": max_train_steps,
    "lora_rank": lora_rank,
    "batch_size": train_batch_size
})

# ================================
# 5. Цикл дообучения
# ================================
while global_step < max_train_steps:
    for batch in dataloader:
        if global_step >= max_train_steps:
            break
            
        pixel_values = batch.to("cuda", dtype=weight_dtype)
        
        # --- ОТВЕТЫ НА ВОПРОСЫ В КОММЕНТАРИЯХ ---
        
        # (1) Зачем нужен VAE и что он делает?
        # VAE (Variational AutoEncoder) сжимает большое изображение (512x512) 
        # в латентное пространство (latent space) меньшего размера (64x64).
        # Благодаря этому диффузия (unet) работает значительно быстрее и требует 
        # меньше VRAM. scaling_factor нужен для масштабирования дисперсии в ~1.0
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor
        
        # (2) Как зашумляется изображение?
        # В латентное представление подмешивается случайно сгенерированный 
        # шум (стандартное Гауссовское распределение) из той же размерности.
        noise = torch.randn_like(latents)
        
        # (3) Как семплируется диффузионное время?
        # Выбирается случайный или псевдослучайный шаг (timestep) от 0 до 999 
        # для каждого элемента батча. Это позволяет модели выучить "распознавание
        # шума" сразу на всех уровнях зашумленности.
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,), 
            device=latents.device
        )
        timesteps = timesteps.long()
        
        # Применяем forward diffusion (добавление шума к латентам в зависимости от timestep)
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
        
        # (4) Что идёт на вход основной модели (UNet)?
        # 1. Зашумленные латентные тензоры (noisy_latents)
        # 2. Текущие временные шаги зашумления (timesteps)
        # 3. Условие — текстовый эмбеддинг для "<cheburashka> plushie", чтобы направить генерацию
        encoder_hidden_states = prompt_embeds.repeat(bsz, 1, 1).to(dtype=weight_dtype)
        
        # Мы используем autocast для вычислений смешанной точности (unet в fp16, lora fp32)
        with torch.autocast("cuda"):
            model_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states
            ).sample
        
        # (5) Какой лосс считается?
        # Для SD1.5 предсказывается "epsilon" (сам первоначальный добавленный шум).
        # Поэтому лоссом является среднеквадратичное отклонение (MSE) между предсказанием 
        # модели и случайным тензором "noise".
        
        # (6) Какие дополнения нужно сделать для этой модели при подсчете лосса?
        # Для базового SD1.5, где prediction_type="epsilon", обычный MSE идеален.
        # В качестве улучшений для дообучения часто применяют взвешивание Min-SNR 
        # (snr_gamma=5.0), которое снижает влияние лосса на сильно зашумленных шагах
        # для стабилизации, или предсказание скорости (v-prediction, в SD2.0).
        # В нашем базовом скрипте используется чистый MSE.
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
        
        # Обратное распространение ошибки (градиент пойдет только в веса LoRA)
        loss.backward()
        
        # Ограничение "взрыва_градиентов" (clipping)
        torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        
        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        global_step += 1
        
        # Отправляем метрику потерь (loss) на график в MLflow
        mlflow.log_metric("loss", loss.item(), step=global_step)
        
        # Сохраняем промежуточные чекпоинты (каждые 200 шагов)
        if global_step % 200 == 0:
            checkpoint_path = f"cheburashka_lora_checkpoint_{global_step}"
            unet.save_pretrained(checkpoint_path)
            print(f"\n[!] Чекпоинт сохранен: {checkpoint_path}")

print("Обучение завершено!")
unet.save_pretrained("cheburashka_lora_final")
print("Финальная LoRA-модель успешно сохранена в папку 'cheburashka_lora_final'")

# Закрываем сессию MLflow
mlflow.end_run()
