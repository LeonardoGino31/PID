# PID — Guía del Proyecto para la Supervisora

**Proyecto:** Physics-Informed Diffusion Model for Infrared Image Generation (PID)  
**Adaptación:** Dataset M3FD (Multi-Modal Multi-Spectral Fusion Dataset)  
**Hardware local:** NVIDIA RTX 4050 · 6 GB VRAM · Python 3.10  
**Fecha:** Junio 2026

---

## 1. Contexto y Objetivo

El modelo PID genera imágenes infrarrojas (IR) a partir de imágenes visibles (RGB) utilizando un modelo de difusión latente condicionado.  El repositorio original fue entrenado en los datasets KAIST, FLIR y VEDAI.  Esta adaptación extiende el modelo al dataset **M3FD**, que contiene pares RGB/IR de escenas urbanas con anotaciones de seis clases: `People`, `Car`, `Bus`, `Motorcycle`, `Lamp`, `Truck`.

**Pregunta de investigación:** ¿Puede el modelo PID, pre-entrenado en KAIST, mejorar sus métricas PSNR y SSIM en M3FD con fine-tuning, comparado con el baseline sin fine-tuning?

**Métricas baseline** (modelo KAIST sin ningún fine-tuning, evaluado en M3FD):

| Métrica | Valor |
|---------|-------|
| PSNR    | 11.29 dB |
| SSIM    | 0.393 |
| Ratio brillo | 0.37 |

---

## 2. Estructura del Repositorio

```
PID/
├── configs/
│   └── latent-diffusion/
│       ├── PID-KAIST-c=M.yaml     # config original (KAIST)
│       └── PID-M3FD-c=M.yaml      # ← NUEVO: config para M3FD
│
├── data/
│   └── M3FD/
│       ├── train.txt              # ← generado por scripts/split_M3FD.py
│       ├── val.txt
│       └── test.txt
│
├── ldm/
│   └── data/
│       ├── KAIST.py               # dataset original (parchado)
│       ├── FLIRv1.py              # dataset original (parchado)
│       ├── vedai512.py            # dataset original (parchado)
│       └── M3FD.py                # ← NUEVO: dataset M3FD
│
├── pretrained/                    # pesos pre-entrenados (descargar de HuggingFace)
│   ├── vqf8_pretrained/
│   │   └── model.ckpt             # pesos del VQGAN
│   └── TeVNet_KAIST/
│       └── epoch_950.pth          # pesos de TeVNet (baseline KAIST)
│
├── scripts/
│   ├── split_M3FD.py              # ← NUEVO: genera train/val/test.txt (seed=42)
│   └── make_synthetic_M3FD.py     # ← NUEVO: mini-dataset sintético para pruebas
│
├── M3FD/                          # dataset real (no incluido en git)
│   ├── ir/                        # imágenes infrarrojas
│   ├── vi/                        # imágenes visibles RGB
│   └── labels/                    # anotaciones XML (opcionales)
│
├── GUIA_SUPERVISORA.md            # ← este documento
└── README.md                      # README original del paper
```

---

## 3. Pesos Pre-entrenados

Los pesos están en HuggingFace: https://huggingface.co/FerrisMao/PID

> **Nota técnica:** El servidor de VQGAN en `ommer-lab.com` está caído. Los pesos del VQGAN (`vqf8_pretrained/model.ckpt`) se extraen directamente del checkpoint PID completo.

### Descarga con huggingface-cli

```bash
pip install huggingface-hub
huggingface-cli download FerrisMao/PID --local-dir pretrained
```

O manualmente desde la página web, colocar los archivos en:
- `pretrained/vqf8_pretrained/model.ckpt`
- `pretrained/TeVNet_KAIST/epoch_950.pth`
- `pretrained/PID_KAIST/last.ckpt` (checkpoint de difusión pre-entrenado)

---

## 4. Instalación del Entorno

```bash
# Crear entorno conda (Python 3.10 obligatorio — pytorch-lightning 1.4.2 no soporta 3.12)
conda create -n pid python=3.10
conda activate pid

# PyTorch con CUDA 12.x (RTX 4050)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Dependencias del proyecto
pip install pytorch-lightning==1.4.2
pip install omegaconf einops transformers kornia
pip install "numpy<2.0"          # numpy 2.x rompe compatibilidad con pytorch-lightning 1.4.2
pip install "pillow>=9.5"
pip install segmentation-models-pytorch  # para TeVNet
pip install tqdm imageio
```

---

## 5. Preparar el Dataset M3FD

### 5a. Con el dataset real

1. Descargar M3FD (contactar a los autores del paper o usar el enlace oficial).
2. Colocar las carpetas `ir/`, `vi/`, `labels/` dentro de `M3FD/`.
3. Generar el split:

```bash
python scripts/split_M3FD.py --m3fd_root M3FD --out_dir data/M3FD
# Salida: data/M3FD/train.txt, val.txt, test.txt (seed=42 → reproducible)
```

### 5b. Con el mini-dataset sintético (para desarrollo/pruebas)

Si aún no tienes el dataset real, puedes generar 60 pares sintéticos:

```bash
python scripts/make_synthetic_M3FD.py --out_root M3FD --n_images 60 --size 512
python scripts/split_M3FD.py --m3fd_root M3FD --out_dir data/M3FD
```

Esto crea imágenes simples de escenas nocturnas simuladas — sólo para verificar que el pipeline de datos funciona correctamente.

---

## 6. Entrenamiento (Fine-tuning)

```bash
python main.py \
    -t True \
    -b configs/latent-diffusion/PID-M3FD-c=M.yaml \
    --gpus 0, \
    model.params.ckpt_path=pretrained/PID_KAIST/last.ckpt
```

**Parámetros importantes para RTX 4050 (6 GB VRAM):**
- `batch_size: 4` — ya configurado en el YAML
- `num_workers: 2` — optimizado para Windows
- Si hay OOM, reducir a `batch_size: 2` o añadir `--accumulate_grad_batches 2`

---

## 7. Split Train/Val/Test (Reproducibilidad)

| Conjunto | Proporción | Semilla |
|----------|-----------|---------|
| Train    | 70%       | 42      |
| Val      | 15%       | 42      |
| Test     | 15%       | 42      |

El script `split_M3FD.py` siempre produce el mismo split con `--seed 42`, garantizando reproducibilidad para publicación.

---

## 8. Parches de Compatibilidad Aplicados

Los siguientes archivos fueron modificados para compatibilidad con Python 3.10 + Pillow ≥ 9.x + NumPy < 2.0:

| Archivo | Problema | Solución |
|---------|----------|----------|
| `ldm/data/KAIST.py` | `PIL.Image.BICUBIC` deprecado | → `PIL.Image.Resampling.BICUBIC` |
| `ldm/data/FLIRv1.py` | Ídem | Ídem |
| `ldm/data/vedai512.py` | Ídem | Ídem |
| `ldm/modules/image_degradation/utils_image.py` | `np.int` deprecado | → `np.int64` |
| `main.py` | DataLoader en Windows | Añadido `persistent_workers` |

---

## 9. Descripción Técnica del Modelo PID

El modelo PID (Physics-Informed Diffusion) combina tres componentes:

1. **VQGAN (VQModelInterface):** autoencoder que comprime imágenes 512×512 a latentes 64×64×4. Entrenado previamente, sus pesos se mantienen fijos durante el fine-tuning.

2. **U-Net de difusión latente:** el corazón del modelo. Recibe como entrada los latentes IR ruidosos concatenados con el condicional RGB (downsampled). Aprende a denoiser la imagen IR condicionado en la imagen visible.

3. **TeVNet (HADARNet):** red auxiliar que extrae información de texturas y contornos de la imagen RGB para guiar el proceso de difusión. Usa una arquitectura Unet con encoder ResNet-18.

**Modo c=M (concat):** el condicional RGB se concatena directamente en el espacio latente al canal de entrada del U-Net. Esto da `in_channels=7` (4 latentes IR + 3 RGB downsampled).

---

## 10. Métricas de Evaluación

Las métricas se calculan con los scripts en `metric/`:

```bash
# Ejemplo: evaluar PSNR y SSIM en imágenes generadas vs. ground truth
python metric/compute_metrics.py \
    --pred_dir logs/PID_M3FD/images/val \
    --gt_dir M3FD/ir
```

**Objetivo de mejora** respecto al baseline (PSNR=11.29, SSIM=0.393):
- PSNR > 13 dB (mejora de ≥ 2 dB considerada significativa)
- SSIM > 0.45

---

## 11. Referencia del Paper Original

```bibtex
@article{PID2023,
  title   = {Physics-Informed Diffusion Model for Infrared Image Generation},
  author  = {Mao, Fanyi and ...},
  journal = {...},
  year    = {2023},
}
```

El README original del repositorio contiene la cita completa.
