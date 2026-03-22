# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from PIL import Image
import numpy as np


class SquarePadPIL:
    def __call__(self, image: Image.Image):
        w, h = image.size
        max_wh = max(w, h)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2
        pad_left = hp
        pad_right = max_wh - w - hp
        pad_top = vp
        pad_bottom = max_wh - h - vp
        padding = (pad_left, pad_top, pad_right, pad_bottom)  # (left, top, right, bottom)
        return TF.pad(image, padding, fill=0, padding_mode='constant')


class SquarePadTensor:
    def __call__(self, image: torch.Tensor):
        """
        Pads a (C, H, W) tensor image to square shape.
        """
        _, h, w = image.shape
        max_wh = max(h, w)
        hp = (max_wh - w) // 2
        vp = (max_wh - h) // 2

        pad_left = hp
        pad_right = max_wh - w - hp
        pad_top = vp
        pad_bottom = max_wh - h - vp
        padding = (pad_left, pad_right, pad_top, pad_bottom)

        # padding = (hp, hp, vp, vp)
        return F.pad(image, padding, value=0)


class NormalizeDepth:
    # ToTensor() expects a numpy array in the range [0, 1] when
    # the array is not uint8.
    def __call__(self, depth_map, norm_max=1.0, norm_min=-1.0, clip=True):
        valid_mask = depth_map != torch.inf
        norm_range = norm_max - norm_min
        quantile_98 = torch.quantile(depth_map[valid_mask], 0.98)
        quantile_02 = torch.quantile(depth_map[valid_mask], 0.02)
        depth_map = ((depth_map - quantile_02) / (quantile_98 - quantile_02)) * norm_range + norm_min
        if clip:
            depth_map = torch.clip(depth_map, norm_min, norm_max)
        return depth_map
        

class NormalizeNp:
    # ToTensor() expects a numpy array in the range [0, 1] when
    # the array is not uint8.
    def __call__(self, image):
        return image / 255.0
        

class ConvertToRGB:
    """Converts PIL image to RGB format. This class is picklable unlike lambda functions."""
    def __call__(self, img):
        return img.convert("RGB")


class RepeatTensor:
    def __init__(self, n):
        self.n = n

    def __call__(self, image: torch.Tensor):
        return image.repeat(self.n, 1, 1)

TRANSFORMS = dict()


def register_transform(transform):
    name = transform.__name__
    if name in TRANSFORMS:
        raise RuntimeError(f"Transform {name} has already registered.")
    TRANSFORMS.update({name: transform})


def get_transform(type, resolution):
    transform = TRANSFORMS[type](resolution)
    transform = T.Compose(transform)
    transform.image_size = resolution
    return transform


@register_transform
def default_train(n_px):
    transform = [
        ConvertToRGB(),
        T.Resize(n_px),  # Image.BICUBIC
        T.CenterCrop(n_px),
        # T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
    return transform


@register_transform
def default_train_from_np(n_px):
    transform = [
        NormalizeNp(),
        T.ToTensor(),
        T.ConvertImageDtype(torch.float32),
        T.Resize(n_px),  # Image.BICUBIC
        T.CenterCrop(n_px),
        T.Normalize([0.5], [0.5]),
    ]
    return transform

