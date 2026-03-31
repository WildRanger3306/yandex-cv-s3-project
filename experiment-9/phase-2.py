"""
Experiment 7 — phase-2.py
Улучшение: Captioning (индивидуальные описания) + Text Encoder в цикле обучения.

Ключевые отличия от phase-2-e5.py:
- НЕТ cheburashka_embeds.pt — вместо него читаем .txt файлы рядом с картинками
- CLIPTokenizer + CLIPTextModel загружаются в память и кодируют текст прямо в цикле
- Каждая картинка получает свой уникальный эмбеддинг при каждом шаге
- Это фундаментально решает проблему "выжигания" фона
"""

import os
import torch
import torch.nn.functional as F
import mlflow
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
from diffusers import DDPMScheduler, UNet2DConditionModel, AutoencoderKL, StableDiffusionPipeline
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPProcessor, CLIPVisionModel

# ================================
# 1. Датасет с Captioning
# ================================
class CheburashkaCaptionDataset(Dataset):
    """
    Датасет с поддержкой индивидуальных текстовых описаний (captions).
    Для каждого файла cheburashka_N.png ищет файл cheburashka_N.txt рядом.
    Если .txt не найден — использует базовый промпт как фоллбэк.
    """
    def __init__(self, data_dir, tokenizer, fallback_prompt="<cheburashka> plushie"):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.fallback_prompt = fallback_prompt
        
        self.image_paths = sorted([
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.lower().endswith('.png')
        ])
        
        self.transform = T.Compose([
            T.Resize(512),
            T.CenterCrop(512),
            T.RandomHorizontalFlip(p=0.5),  # аугментация
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])
        
        print(f"Датасет загружен: {len(self.image_paths)} картинок")
        self._verify_captions()

    def _verify_captions(self):
        """Проверяет и выводит статистику по наличию .txt файлов"""
        found, missing = 0, 0
        for img_path in self.image_paths:
            txt_path = os.path.splitext(img_path)[0] + ".txt"
            if os.path.exists(txt_path):
                found += 1
            else:
                missing += 1
        print(f"  Найдено captions: {found}/{len(self.image_paths)}")
        if missing:
            print(f"  ⚠️  Без caption (будет использован fallback): {missing} картинок")

    def _get_caption(self, img_path):
        """Читает .txt рядом с картинкой, или возвращает fallback"""
        txt_path = os.path.splitext(img_path)[0] + ".txt"
        if os.path.exists(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        return self.fallback_prompt

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(img)
        
        caption = self._get_caption(img_path)
        
        # Токенизируем текст прямо тут (возвращаем input_ids для batch коллации)
        input_ids = self.tokenizer(
            caption,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze(0)  # убираем размерность батча
        
        return pixel_values, input_ids


def collate_fn(batch):
    """Склеивает батч из (pixel_values, input_ids) пар"""
    pixel_values = torch.stack([item[0] for item in batch])
    input_ids = torch.stack([item[1] for item in batch])
    return pixel_values, input_ids


# ================================
# 2. Загрузка моделей
# ================================
print("Загрузка базовой модели SD 1.5...")
model_id = "runwayml/stable-diffusion-v1-5"
weight_dtype = torch.float16

# Загружаем токенизатор и Text Encoder (нужны для captioning!)
print("Загрузка CLIPTokenizer и CLIPTextModel...")
tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    model_id, subfolder="text_encoder", torch_dtype=weight_dtype
).to("cuda")
text_encoder.requires_grad_(True)  # Размораживаем для Fine-tuning!

noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype).to("cuda")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=weight_dtype).to("cuda")

vae.requires_grad_(False)
unet.requires_grad_(False)

if is_xformers_available():
    print("Включение xFormers для ускорения...")
    unet.enable_xformers_memory_efficient_attention()

# --- ИНСТРУМЕНТАРИЙ ДЛЯ МЕТРИК ---
print("Загрузка CLIP Vision для метрики сходства...")
metric_model_id = "openai/clip-vit-base-patch32"
metric_processor = CLIPProcessor.from_pretrained(metric_model_id)
metric_vision_model = CLIPVisionModel.from_pretrained(metric_model_id).to("cuda", dtype=weight_dtype)

def get_image_embedding(image):
    """Извлекает визуальный эмбеддинг из картинки"""
    inputs = metric_processor(images=image, return_tensors="pt").to("cuda")
    # Приводим к типу weight_dtype (fp16) для совместимости
    inputs = {k: v.to(dtype=weight_dtype) if v.is_floating_point() else v for k, v in inputs.items()}
    with torch.no_grad():
        outputs = metric_vision_model(**inputs)
    return outputs.last_hidden_state[:, 0, :] # CLS токен

# ================================
# 3. Датасет
# ================================
print("Загрузка датасета с captions...")
train_batch_size = 1
dataset = CheburashkaCaptionDataset(
    data_dir="../dataset-2",
    tokenizer=tokenizer,
)
dataloader = DataLoader(
    dataset,
    batch_size=train_batch_size,
    shuffle=True,
    collate_fn=collate_fn
)

# ================================
# 4. Добавление LoRA
# ================================
print("Инициализация адаптеров LoRA (PEFT)...")
lora_rank = 16
lora_config = LoraConfig(
    r=lora_rank,
    lora_alpha=16,
    lora_dropout=0.1,
    init_lora_weights="gaussian",
    # Captioning позволяет безопасно использовать все 4 модуля:
    # модель сама учится разграничивать персонажа и фон через текст
    target_modules=["to_k", "to_q", "to_v", "to_out.0"]
)

unet.add_adapter(lora_config)

# LoRA веса в fp32 для стабильности обучения
for param in unet.parameters():
    if param.requires_grad:
        param.data = param.data.to(torch.float32)

# ================================
# 5. Настройка оптимизатора
# ================================
learning_rate = 4e-5
learning_rate_text = 1e-5  # Более низкий LR для текстового энкодера
max_train_steps = 1000
max_grad_norm = 1.0
gradient_accumulation_steps = 4
weight_decay = 1e-2
noise_offset = 0.1  # Для улучшения контраста

optimizer = torch.optim.AdamW([
    {"params": filter(lambda p: p.requires_grad, unet.parameters()), "lr": learning_rate},
    {"params": filter(lambda p: p.requires_grad, text_encoder.parameters()), "lr": learning_rate_text},
], weight_decay=weight_decay)

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=max_train_steps,
)

scaler = torch.amp.GradScaler("cuda")

unet.train()
global_step = 0
inner_step = 0
progress_bar = tqdm(total=max_train_steps, desc="Обучение (Experiment-9 Dead, Sketch and Bike)")

# ================================
# 6. MLflow
# ================================
mlflow.set_tracking_uri("http://188.243.201.66:5000")
mlflow.set_experiment("cheburashka-lora-experiment-9")
mlflow.start_run()
mlflow.log_params({
    "experiment": 8,
    "captioning": True,
    "learning_rate_unet": learning_rate,
    "learning_rate_text": learning_rate_text,
    "noise_offset": noise_offset,
    "max_train_steps": max_train_steps,
    "lora_rank": lora_rank,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "batch_size": train_batch_size,
    "gradient_accumulation_steps": gradient_accumulation_steps,
    "lr_scheduler": "cosine",
    "snr_gamma": 5.0,
    "weight_decay": weight_decay,
    "target_modules": "to_k, to_q, to_v, to_out.0",
    "metrics_version": "1.1 (Similarity + GradNorm)",
})

# Подготовим референсный эмбеддинг Чебурашки из датасета (для метрики Similarity)
print("[*] Фиксация эталона для метрики сходства...")
ref_image = Image.open(dataset.image_paths[0]).convert("RGB")
ref_embedding = get_image_embedding(ref_image)

# ================================
# 7. Цикл обучения
# ================================
while global_step < max_train_steps:
    for pixel_values, input_ids in dataloader:
        if global_step >= max_train_steps:
            break

        inner_step += 1

        # Картинки → GPU
        pixel_values = pixel_values.to("cuda", dtype=weight_dtype)
        
        # ============================================================
        # КЛЮЧЕВОЕ ОТЛИЧИЕ ОТ ПРЕДЫДУЩИХ ВЕРСИЙ:
        # Кодируем текст прямо в цикле обучения — для КАЖДОЙ картинки
        # свой уникальный эмбеддинг, отражающий её описание.
        # ============================================================
        input_ids = input_ids.to("cuda")
        with torch.no_grad():
            encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=weight_dtype)

        # Кодируем изображение через VAE
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Добавляем шум с Noise Offset
        noise = torch.randn_like(latents)
        if noise_offset:
            # Noise offset позволяет модели учить очень темные и светлые участки
            noise += noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1), device=latents.device)
            
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (bsz,),
            device=latents.device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Forward pass UNet
        with torch.autocast("cuda"):
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # Min-SNR Loss
        loss = F.mse_loss(model_pred.float(), noise.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape))))

        alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
        alpha_t = alphas_cumprod[timesteps] ** 0.5
        sigma_t = (1.0 - alphas_cumprod[timesteps]) ** 0.5
        snr = (alpha_t / sigma_t) ** 2
        snr_weight = torch.clamp(snr, max=5.0) / snr

        loss = (loss * snr_weight).mean()
        loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        # Optimizer step раз в 4 батча (gradient accumulation)
        if inner_step % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            
            # --- МЕТРИКА 1: Gradient Norm (логируем до клиппинга) ---
            params_to_clip = list(unet.parameters()) + list(text_encoder.parameters())
            grad_norm = torch.nn.utils.clip_grad_norm_(params_to_clip, max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad()

            current_lr = lr_scheduler.get_last_lr()[0]
            display_loss = loss.item() * gradient_accumulation_steps
            progress_bar.update(1)
            progress_bar.set_postfix(loss=f"{display_loss:.4f}", lr=f"{current_lr:.6f}", gn=f"{grad_norm:.2f}")
            global_step += 1

            mlflow.log_metric("loss", display_loss, step=global_step)
            mlflow.log_metric("lr", current_lr, step=global_step)
            mlflow.log_metric("grad_norm", grad_norm.item(), step=global_step)

            if global_step % 200 == 0:
                checkpoint_path = f"cheburashka_lora_checkpoint_{global_step}"
                os.makedirs(checkpoint_path, exist_ok=True)
                
                # Сохраняем LoRA веса UNet
                lora_state_dict = {k: v.cpu() for k, v in unet.state_dict().items() if "lora" in k}
                torch.save(lora_state_dict, os.path.join(checkpoint_path, "lora_weights.pt"))
                
                # Сохраняем Fine-tuned Text Encoder (полностью, так как он маленький)
                text_encoder.save_pretrained(os.path.join(checkpoint_path, "text_encoder"))
                
                print(f"\n[!] Чекпоинт сохранен: {checkpoint_path}")

                # ============================================================
                # ВАЛИДАЦИЯ (Генерация проверочной сетки)
                # ============================================================
                print(f"[*] Генерирую валидационные изображения на шаге {global_step}...")
                validation_prompts = [
                    "<cheburashka> plushie",
                    "<cheburashka> with the Eiffel Tower in the background",
                    "<cheburashka> in sketch style",
                    "<cheburashka> riding a bicycle"
                ]
                
                unet.eval()
                text_encoder.eval()
                
                with torch.no_grad():
                    val_pipeline = StableDiffusionPipeline.from_pretrained(
                        model_id,
                        unet=unet,
                        text_encoder=text_encoder,
                        vae=vae,
                        tokenizer=tokenizer,
                        torch_dtype=weight_dtype,
                        safety_checker=None
                    ).to("cuda")
                    val_pipeline.set_progress_bar_config(disable=True)
                    
                    images = []
                    for prompt in validation_prompts:
                        image = val_pipeline(prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
                        images.append(image)
                    
                    # Собираем в сетку 2x2
                    grid_img = Image.new('RGB', (1024, 1024))
                    grid_img.paste(images[0], (0, 0))
                    grid_img.paste(images[1], (512, 0))
                    grid_img.paste(images[2], (0, 512))
                    grid_img.paste(images[3], (512, 512))
                    
                    # --- МЕТРИКА 2: Identity Similarity (CLIP) ---
                    similarities = []
                    for val_img in images:
                        val_emb = get_image_embedding(val_img)
                        # Косинусное сходство между валидацией и эталоном
                        sim = torch.nn.functional.cosine_similarity(ref_embedding, val_emb).item()
                        similarities.append(sim)
                    
                    avg_similarity = sum(similarities) / len(similarities)
                    mlflow.log_metric("identity_similarity", avg_similarity, step=global_step)
                    print(f"[#] Сходство с оригиналом: {avg_similarity:.4f}")
                    
                    mlflow.log_image(grid_img, f"val_{global_step}.png")
                    print(f"[#] Валидация успешна, изображения загружены в MLflow")
                
                # Возвращаем в режим обучения
                del val_pipeline
                torch.cuda.empty_cache()
                unet.train()
                text_encoder.train()

print("Обучение завершено!")
final_path = "cheburashka_lora_final"
os.makedirs(final_path, exist_ok=True)
lora_state_dict = {k: v.cpu() for k, v in unet.state_dict().items() if "lora" in k}
torch.save(lora_state_dict, os.path.join(final_path, "lora_weights.pt"))
print("Финальная LoRA-модель сохранена в 'cheburashka_lora_final'")

mlflow.end_run()
