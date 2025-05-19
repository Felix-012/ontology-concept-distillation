import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset


class DatasetGenerator(Dataset):
    def __init__(self, basedir, file_names, labels, transform, use_cache=False):
        """
        Args:
            basedir (str): Base directory containing the images
            file_names (str): Path to dataset file OR list of image paths
            transform: Transforms to apply to images
            use_cache (bool): Whether to cache images in memory
        """
        self.basedir = basedir

        # Assume file_names is list of image paths and labels are provided
        self.listImagePaths = [os.path.join(basedir, img_path)[:-4] + ".jpg" for img_path in file_names]
        self.listImageLabels = labels

        self.transform = transform
        self.use_cache = use_cache
        self.image_cache = {}  # Cache to store loaded images
        self.replace_with_png = False

    def check_jpg_vs_png(self, image_path):
        if os.path.exists(image_path):
            return False
        if os.path.exists(image_path[:-4] + ".png"):
            return True
        raise FileNotFoundError(f"File not found (also not the PNG version): {image_path} or {image_path[:-4] + '.png'}")

    def load_image(self, image_path):
        if image_path not in self.image_cache:
            img = Image.open(image_path).convert('RGB').resize((512, 512))
            if self.use_cache == True: 
                self.image_cache[image_path] = img
            else: 
                return img
        return self.image_cache[image_path]

    def __getitem__(self, index):
        imagePath = self.listImagePaths[index]

        # Check file extension replacement only once
        if index == 0:
            self.replace_with_png = self.check_jpg_vs_png(imagePath)

        if self.replace_with_png:
            imagePath = imagePath[:-4] + ".png"

        imageData = self.load_image(imagePath)
        imageLabel = torch.FloatTensor(self.listImageLabels[index])

        if self.transform:
            imageData = self.transform(imageData)

        return imageData, imageLabel

    def __len__(self):
        return len(self.listImagePaths)
