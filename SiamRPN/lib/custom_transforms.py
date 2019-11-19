import torch
import numpy as np
import cv2
from net.config import Config
from torchvision.transforms import RandomCrop


class RandomStretch:
    def __init__(self, max_stretch=0.15):
        self.max_stretch = Config.scale_resize

    def __call__(self, img):
        random_scaled_h = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        random_scaled_w = 1.0 + np.random.uniform(-self.max_stretch, self.max_stretch)
        img_h, img_w, _ = img.shape
        new_shape = img_h * random_scaled_h, img_w * random_scaled_w
        return cv2.resize(img, new_shape, cv2.INTER_LINEAR)


class CenterCrop:
    def __init__(self, crop_size=Config.instance_size):
        self.instance_size = crop_size

    def __call__(self, img):
        pass


class RandomCrop():
    pass


class ColorAug:
    pass


class RandomBlur:
    pass


class Normalize:
    pass


class ToTensor:
    def __init__(self):
        pass

    def __call__(self, img):
        return torch.from_numpy(img.astype(np.float32))
