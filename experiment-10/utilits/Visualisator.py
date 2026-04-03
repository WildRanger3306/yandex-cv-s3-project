import os

import cv2
from PIL import Image
import matplotlib.pyplot as plt

class Visualisator:
    def __init__(self, images_paths: str, save_path: str):
        self.images_path = images_paths
        self.save_path = save_path
        files = os.listdir(images_paths)
        self.image_paths = [
            os.path.join(images_paths, img) 
            for img in files 
            if os.path.splitext(img)[1] in ['.png', '.jpg']
            ]

    def visualize(self, title: str, save_img_name: str):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plt.suptitle(title, y=0.9)
        for i, path in enumerate(self.image_paths[:3]):
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img)
            axes[i].axis('off')
        plt.savefig(os.path.join(self.save_path, save_img_name))
        plt.show()
