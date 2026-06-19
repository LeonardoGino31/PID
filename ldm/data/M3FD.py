"""
Dataset M3FD (Multi-Modal Multi-Spectral Fusion Dataset) para PID.

Estructura esperada en disco:
    M3FD/
    ├── ir/          # imágenes infrarrojas  (*.png)
    ├── vi/          # imágenes visibles RGB  (*.png)
    └── labels/      # anotaciones XML PASCAL-VOC (opcional, no usadas en entrenamiento)

Los splits se leen desde archivos .txt generados por scripts/split_M3FD.py:
    data/M3FD/train.txt
    data/M3FD/val.txt
    data/M3FD/test.txt

Cada línea del .txt contiene el nombre de archivo sin extensión, p.ej.:
    00001
    00002
    ...

Clases presentes: People, Car, Bus, Motorcycle, Lamp, Truck.
Las etiquetas XML no se cargan aquí; sólo se usan las imágenes emparejadas.

Compatibilidad: Python 3.10, Pillow ≥ 9.x (sin PIL.Image.LINEAR/BICUBIC legacy).
"""

import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
import random
import torchvision.transforms.functional as tf
from copy import deepcopy


# ---------------------------------------------------------------------------
# Constante de interpolación compatible con Pillow ≥ 9.1
# ---------------------------------------------------------------------------
_INTERP = {
    "linear":   Image.Resampling.BILINEAR,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic":  Image.Resampling.BICUBIC,
    "lanczos":  Image.Resampling.LANCZOS,
}


# ---------------------------------------------------------------------------
# Augmentaciones pareadas
# ---------------------------------------------------------------------------

def _random_crop_pair(img1: Image.Image, img2: Image.Image, min_ratio=0.5):
    w, h = img1.size
    scale = min_ratio + random.random() * (1.0 - min_ratio)
    nw, nh = int(w * scale), int(h * scale)
    x = random.randint(0, w - nw)
    y = random.randint(0, h - nh)
    return img1.crop((x, y, x + nw, y + nh)), img2.crop((x, y, x + nw, y + nh))


class M3FDBase(Dataset):
    """
    Dataset base para M3FD.

    Parámetros
    ----------
    txt_file : str
        Ruta al archivo .txt con los nombres de imagen (sin extensión).
    data_root : str
        Raíz del dataset M3FD. Debe contener subcarpetas 'ir/' y 'vi/'.
    size : int | None
        Tamaño de salida (cuadrado). Si es None no se redimensiona.
    interpolation : str
        Método de interpolación: 'bicubic' (defecto), 'bilinear', 'lanczos'.
    flip_p : float
        Probabilidad de flip horizontal en train.
    crop_p : float
        Probabilidad de random crop en train.
    """

    def __init__(
        self,
        txt_file: str,
        data_root: str,
        size: int = None,
        interpolation: str = "bicubic",
        flip_p: float = 0.5,
        crop_p: float = 0.5,
    ):
        self.data_root = data_root
        self.ir_root = os.path.join(data_root, "ir")
        self.vi_root = os.path.join(data_root, "vi")

        with open(txt_file, "r") as f:
            self.image_names = [l.strip() for l in f if l.strip()]

        self._length = len(self.image_names)
        self.size = size
        self.interp = _INTERP.get(interpolation, Image.Resampling.BICUBIC)
        self.flip_p = flip_p
        self.crop_p = crop_p

    def __len__(self):
        return self._length

    def _load_pair(self, name: str):
        ir_path = os.path.join(self.ir_root, name + ".png")
        vi_path = os.path.join(self.vi_root, name + ".png")

        img_ir = Image.open(ir_path).convert("RGB")
        img_vi = Image.open(vi_path).convert("RGB")
        return img_ir, img_vi

    @staticmethod
    def _center_crop_square(img: Image.Image) -> Image.Image:
        w, h = img.size
        s = min(w, h)
        left = (w - s) // 2
        top  = (h - s) // 2
        return img.crop((left, top, left + s, top + s))

    def __getitem__(self, i: int):
        name = self.image_names[i]
        img_ir, img_vi = self._load_pair(name)

        # Recorte cuadrado centrado
        img_ir = self._center_crop_square(img_ir)
        img_vi = self._center_crop_square(img_vi)

        # Augmentaciones (sólo relevantes en train; val/test las deshabilitan con p=0)
        if self.crop_p > 0 and random.random() < self.crop_p:
            img_ir, img_vi = _random_crop_pair(img_ir, img_vi)

        if self.size is not None:
            img_ir = img_ir.resize((self.size, self.size), resample=self.interp)
            img_vi = img_vi.resize((self.size, self.size), resample=self.interp)

        arr_ir = np.array(img_ir, dtype=np.uint8)
        arr_vi = np.array(img_vi, dtype=np.uint8)

        if self.flip_p > 0 and random.random() < self.flip_p:
            arr_ir = np.flip(arr_ir, axis=1).copy()
            arr_vi = np.flip(arr_vi, axis=1).copy()

        # Normalización a [-1, 1] como el resto de datasets PID
        return {
            "image":       (arr_ir / 127.5 - 1.0).astype(np.float32),
            "conditional": (arr_vi / 127.5 - 1.0).astype(np.float32),
            "file_path_":  name,
        }


class M3FDTrain(M3FDBase):
    def __init__(self, **kwargs):
        super().__init__(
            txt_file="data/M3FD/train.txt",
            data_root="M3FD",
            **kwargs,
        )


class M3FDVal(M3FDBase):
    def __init__(self, flip_p=0.0, crop_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/M3FD/val.txt",
            data_root="M3FD",
            flip_p=flip_p,
            crop_p=crop_p,
            **kwargs,
        )


class M3FDTest(M3FDBase):
    def __init__(self, flip_p=0.0, crop_p=0.0, **kwargs):
        super().__init__(
            txt_file="data/M3FD/test.txt",
            data_root="M3FD",
            flip_p=flip_p,
            crop_p=crop_p,
            **kwargs,
        )
