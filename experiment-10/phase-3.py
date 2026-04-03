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

# Создадим снова пайплайн генерации и загрузим обученную модель.
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cuda")

pipe.load_lora_weights(final_save_path)

pipe.safety_checker = None
pipe.requires_safety_checker = False


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
image


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
            height=1024,
            width=1024,
            guidance_scale=7.5
        ).images[0]

        path_to_save = os.path.join(folder_to_save, f'cheburashka_{n}.png')
        image.save(path_to_save)

        del image
        gc.collect()
        torch.cuda.empty_cache() 
        torch.cuda.synchronize()