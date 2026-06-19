"""
split_M3FD.py — Split reproducible del dataset M3FD en train/val/test.

Uso:
    python scripts/split_M3FD.py --m3fd_root M3FD --out_dir data/M3FD

Parámetros
----------
--m3fd_root : str
    Carpeta raíz del dataset M3FD (debe contener subcarpeta 'ir/').
--out_dir : str
    Carpeta de salida para train.txt, val.txt, test.txt.
--train_ratio : float (default 0.70)
--val_ratio   : float (default 0.15)
--seed        : int   (default 42)

El test set es el complemento: test_ratio = 1 - train_ratio - val_ratio.

Proporciones usadas en la investigación (seed=42):
    70% train / 15% val / 15% test

Métricas baseline (modelo KAIST sin fine-tuning en M3FD):
    PSNR=11.29 dB | SSIM=0.393 | ratio_brillo=0.37
"""

import argparse
import os
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Split M3FD dataset reproducibly.")
    p.add_argument("--m3fd_root",  default="M3FD",       help="Raíz del dataset M3FD")
    p.add_argument("--out_dir",    default="data/M3FD",   help="Carpeta de salida para los .txt")
    p.add_argument("--train_ratio",type=float, default=0.70)
    p.add_argument("--val_ratio",  type=float, default=0.15)
    p.add_argument("--seed",       type=int,   default=42)
    return p.parse_args()


def main():
    args = parse_args()

    ir_dir = Path(args.m3fd_root) / "ir"
    if not ir_dir.exists():
        raise FileNotFoundError(
            f"No se encontró la carpeta de IR: {ir_dir}\n"
            "Asegúrate de que el dataset M3FD está en la ruta correcta."
        )

    # Recoger todos los stems (sin extensión) de las imágenes IR
    names = sorted(p.stem for p in ir_dir.glob("*.png"))
    if not names:
        names = sorted(p.stem for p in ir_dir.glob("*.jpg"))
    if not names:
        raise RuntimeError(f"No se encontraron imágenes .png/.jpg en {ir_dir}")

    total = len(names)
    print(f"Total de imágenes encontradas: {total}")

    # Split reproducible
    rng = random.Random(args.seed)
    shuffled = names[:]
    rng.shuffle(shuffled)

    n_train = int(total * args.train_ratio)
    n_val   = int(total * args.val_ratio)
    n_test  = total - n_train - n_val

    train = shuffled[:n_train]
    val   = shuffled[n_train:n_train + n_val]
    test  = shuffled[n_train + n_val:]

    print(f"  Train : {len(train):>5}  ({len(train)/total*100:.1f}%)")
    print(f"  Val   : {len(val):>5}  ({len(val)/total*100:.1f}%)")
    print(f"  Test  : {len(test):>5}  ({len(test)/total*100:.1f}%)")

    # Guardar
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
        fpath = out / f"{split_name}.txt"
        fpath.write_text("\n".join(split_list) + "\n", encoding="utf-8")
        print(f"  Guardado: {fpath}")

    print("\nSplit completado. Seed usado:", args.seed)
    print("Para reproducir exactamente este split, siempre usa --seed 42")


if __name__ == "__main__":
    main()
