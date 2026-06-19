"""
extract_vqgan.py — Extrae los pesos del VQGAN (first_stage_model) del checkpoint PID.

El servidor original de VQGAN (ommer-lab.com) está caído.
Los pesos están embebidos dentro de los checkpoints PID completos de HuggingFace.

Uso:
    python scripts/extract_vqgan.py \
        --ckpt pretrained/PID_KAIST/last.ckpt \
        --out  pretrained/vqf8_pretrained/model.ckpt
"""

import argparse
import torch
from pathlib import Path


def extract_vqgan(ckpt_path: str, out_path: str):
    print(f"Cargando checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dict = ckpt.get("state_dict", ckpt)

    # Filtrar sólo las claves del first_stage_model (VQGAN)
    prefix = "first_stage_model."
    vqgan_sd = {
        k[len(prefix):]: v
        for k, v in state_dict.items()
        if k.startswith(prefix)
    }

    if not vqgan_sd:
        print("ERROR: No se encontraron claves 'first_stage_model.*' en el checkpoint.")
        print("Claves disponibles (primeras 20):")
        for k in list(state_dict.keys())[:20]:
            print(f"  {k}")
        return

    print(f"  Encontradas {len(vqgan_sd)} capas del VQGAN.")

    # Guardamos en el formato que espera VQModelInterface (state_dict directo)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": vqgan_sd}, str(out))
    print(f"  VQGAN guardado en: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="pretrained/PID_KAIST/last.ckpt")
    p.add_argument("--out",  default="pretrained/vqf8_pretrained/model.ckpt")
    args = p.parse_args()
    extract_vqgan(args.ckpt, args.out)


if __name__ == "__main__":
    main()
