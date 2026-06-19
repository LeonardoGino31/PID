"""
verify_dataloader.py — Verifica que el pipeline de datos M3FD funciona correctamente.

Carga un batch de train y val, muestra formas y estadísticas.

Uso (desde la raíz del repo PID):
    python scripts/verify_dataloader.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from torch.utils.data import DataLoader
from ldm.data.M3FD import M3FDTrain, M3FDVal


def check_split(name, dataset, batch_size=2):
    print(f"\n{'='*50}")
    print(f"  Split: {name}  ({len(dataset)} imágenes)")
    print(f"{'='*50}")

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    img  = batch["image"]
    cond = batch["conditional"]

    print(f"  image shape      : {list(img.shape)}   dtype={img.dtype}")
    print(f"  conditional shape: {list(cond.shape)}  dtype={cond.dtype}")
    print(f"  image   min/max  : {img.min():.3f} / {img.max():.3f}  (esperado: ~-1 / ~1)")
    print(f"  cond    min/max  : {cond.min():.3f} / {cond.max():.3f}")
    print(f"  file_path_[0]    : {batch['file_path_'][0]}")

    assert img.shape[-1] == 3,  "ERROR: imagen IR no tiene 3 canales"
    assert cond.shape[-1] == 3, "ERROR: condicional no tiene 3 canales"
    assert img.min() >= -1.01 and img.max() <= 1.01, "ERROR: valores fuera de rango [-1,1]"
    print(f"  [OK] Batch verificado correctamente.")


def main():
    print("\nVerificando pipeline de datos M3FD (dataset sintético)...")

    train_ds = M3FDTrain(size=256)   # tamaño reducido para prueba rápida
    val_ds   = M3FDVal(size=256)

    check_split("Train", train_ds)
    check_split("Val",   val_ds)

    print("\n" + "="*50)
    print("  RESULTADO: Pipeline M3FD OK.")
    print("  El dataloader está listo para entrenamiento.")
    print("="*50)


if __name__ == "__main__":
    main()
