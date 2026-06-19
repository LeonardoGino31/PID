"""
make_synthetic_M3FD.py — Genera un mini-dataset M3FD sintético para desarrollo local.

Crea N pares de imágenes RGB/IR sintéticos con texturas simples que simulan
escenas de vigilancia (fondos de ciudad, objetos calientes).  No requiere el
dataset real; sirve para verificar el pipeline de datos antes de bajarlo.

Uso:
    python scripts/make_synthetic_M3FD.py --out_root M3FD --n_images 60

Esto genera:
    M3FD/
    ├── ir/   00001.png … 00060.png   (escala de grises → RGB simulando IR)
    └── vi/   00001.png … 00060.png   (imagen RGB visible sintética)

Luego ejecuta:
    python scripts/split_M3FD.py --m3fd_root M3FD --out_dir data/M3FD

Y el pipeline estará listo para correr con batch_size=1 en CPU o GPU pequeña.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default="M3FD",  help="Raíz de salida del dataset")
    p.add_argument("--n_images", type=int, default=60, help="Número de pares a generar")
    p.add_argument("--size",     type=int, default=512, help="Resolución de cada imagen")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def make_visible_image(rng: random.Random, size: int) -> np.ndarray:
    """Imagen RGB que simula una calle de noche: fondo oscuro + objetos."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    # Fondo: cielo nocturno gris-azulado
    img[:, :] = [rng.randint(20, 40), rng.randint(20, 40), rng.randint(40, 70)]

    # Calzada horizontal
    road_y = int(size * 0.55)
    img[road_y:, :] = [rng.randint(50, 80)] * 3

    # Añadir 3-6 "vehículos" (rectángulos de distintos colores)
    n_cars = rng.randint(3, 6)
    for _ in range(n_cars):
        cx = rng.randint(30, size - 30)
        cy = rng.randint(road_y, size - 20)
        w  = rng.randint(40, 90)
        h  = rng.randint(20, 45)
        color = [rng.randint(80, 200), rng.randint(60, 180), rng.randint(60, 180)]
        x0, y0 = max(0, cx - w // 2), max(0, cy - h // 2)
        x1, y1 = min(size, cx + w // 2), min(size, cy + h // 2)
        img[y0:y1, x0:x1] = color

    # Añadir 2-4 "peatones" (rectángulos estrechos y altos)
    n_people = rng.randint(2, 4)
    for _ in range(n_people):
        cx = rng.randint(20, size - 20)
        cy = rng.randint(road_y - 30, size - 50)
        w  = rng.randint(10, 20)
        h  = rng.randint(40, 80)
        color = [rng.randint(100, 220), rng.randint(80, 200), rng.randint(80, 180)]
        x0, y0 = max(0, cx - w // 2), max(0, cy)
        x1, y1 = min(size, cx + w // 2), min(size, cy + h)
        img[y0:y1, x0:x1] = color

    return img


def make_ir_image(visible: np.ndarray, rng: random.Random) -> np.ndarray:
    """
    Simula una imagen IR a partir de la visible.
    Los objetos brillantes en visible → más calor en IR.
    Fondo frío (~30-50 DN), vehículos/personas calientes (180-240 DN).
    """
    gray = 0.299 * visible[:, :, 0] + 0.587 * visible[:, :, 1] + 0.114 * visible[:, :, 2]

    # Umbral: píxeles brillantes son "calientes"
    hot_mask = gray > 80
    cold_bg  = np.random.RandomState(rng.randint(0, 9999)).randint(20, 60, gray.shape).astype(np.float32)

    ir_channel = cold_bg.copy()
    ir_channel[hot_mask] = gray[hot_mask] * rng.uniform(0.85, 1.05)
    ir_channel = np.clip(ir_channel + rng.gauss(0, 4), 0, 255).astype(np.uint8)

    # IR se presenta como imagen RGB con los 3 canales iguales (claroscuro térmico)
    ir_rgb = np.stack([ir_channel] * 3, axis=-1)
    return ir_rgb


def main():
    args = parse_args()
    rng = random.Random(args.seed)

    ir_dir = Path(args.out_root) / "ir"
    vi_dir = Path(args.out_root) / "vi"
    ir_dir.mkdir(parents=True, exist_ok=True)
    vi_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generando {args.n_images} pares sintéticos en '{args.out_root}/' ...")
    for i in range(1, args.n_images + 1):
        name = f"{i:05d}.png"
        vi_arr = make_visible_image(rng, args.size)
        ir_arr = make_ir_image(vi_arr, rng)

        Image.fromarray(vi_arr).save(vi_dir / name)
        Image.fromarray(ir_arr).save(ir_dir / name)

        if i % 10 == 0:
            print(f"  {i}/{args.n_images} generados...")

    print(f"\nDataset sintético listo: {ir_dir.parent.resolve()}")
    print("Siguiente paso:")
    print("  python scripts/split_M3FD.py --m3fd_root M3FD --out_dir data/M3FD")


if __name__ == "__main__":
    main()
