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

viz = Visualisator(PATH_TO_IMAGES, PATH_TO_ARTIFACTS)
viz.visualize('Чебурашка. Примеры для обучения.', 'cheburashka_for_train.png')

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
)
pipe = pipe.to(device)

prompt = "<cheburashka> with the Eiffel Tower in the background"
negative_prompt = "low quality, blurry"

folder_to_save = os.path.join(PATH_TO_ARTIFACTS, 'before_train')
os.makedirs(folder_to_save, exist_ok=True)
for i in range(3):
    image = pipe(
        prompt, 
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        height=1024,
        width=1024,
        guidance_scale=7.5
    ).images[0]

    path_to_save = os.path.join(folder_to_save, f'cheburashka_raw_{i}.png')
    image.save(path_to_save)

    del image
    gc.collect()
    torch.cuda.empty_cache() 
    torch.cuda.synchronize()

viz = Visualisator(folder_to_save, PATH_TO_ARTIFACTS)
viz.visualize('Чебурашка. Не обученная модель.', 'cheburashka_raw_model.png')

prompt_text = "<cheburashka> plushie"

with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt=prompt_text,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

torch.save({
    "prompt_embeds": prompt_embeds.cpu(),
    "negative_prompt_embeds": negative_prompt_embeds.cpu()
}, "embeddings/cheburashka_plushie_embeddings.pt")

del pipe
gc.collect()
torch.cuda.empty_cache() 
torch.cuda.synchronize()

