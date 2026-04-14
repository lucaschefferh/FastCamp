"""
split_dataset.py
────────────────────────────────────────────────────────────────
Divide o dataset LGG-MRI em train / val / test (70 / 15 / 15 %)
A divisão é feita por PACIENTE para evitar data leakage.

Estrutura esperada:
    archive/kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif
            TCGA_CS_4941_19960909_1_mask.tif
            ...
        ...

Estrutura gerada:
    dataset/
        train/
            images/  *.tif
            masks/   *_mask.tif
        val/
            images/
            masks/
        test/
            images/
            masks/
"""

import os
import shutil
import random
from pathlib import Path

# ─────────────────────────────────────────────────────
# CONFIGURAÇÕES
# ─────────────────────────────────────────────────────
SOURCE_DIR   = Path("archive/kaggle_3m")   # pasta raiz do dataset original
OUTPUT_DIR   = Path("dataset")             # pasta de saída
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
TEST_RATIO   = 0.15
RANDOM_SEED  = 42
# ─────────────────────────────────────────────────────

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9, "Ratios must sum to 1"

random.seed(RANDOM_SEED)

# Coleta apenas diretórios de pacientes (ignora arquivos como data.csv)
patient_dirs = sorted([
    d for d in SOURCE_DIR.iterdir()
    if d.is_dir()
])

n_total   = len(patient_dirs)
n_train   = round(n_total * TRAIN_RATIO)
n_val     = round(n_total * VAL_RATIO)
n_test    = n_total - n_train - n_val   # garante que não fique nenhum de fora

random.shuffle(patient_dirs)

splits = {
    "train": patient_dirs[:n_train],
    "val":   patient_dirs[n_train : n_train + n_val],
    "test":  patient_dirs[n_train + n_val :],
}

print(f"\nTotal de pacientes: {n_total}")
print(f"  train : {len(splits['train'])} pacientes")
print(f"  val   : {len(splits['val'])} pacientes")
print(f"  test  : {len(splits['test'])} pacientes\n")


def copy_patient(patient_dir: Path, images_out: Path, masks_out: Path):
    """Copia todas as imagens e máscaras de um paciente para as pastas de saída."""
    for f in sorted(patient_dir.glob("*.tif")):
        if "_mask" in f.name:
            shutil.copy2(f, masks_out / f.name)
        else:
            shutil.copy2(f, images_out / f.name)


for split_name, patients in splits.items():
    images_dir = OUTPUT_DIR / split_name / "images"
    masks_dir  = OUTPUT_DIR / split_name / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    n_images = 0
    for patient in patients:
        copy_patient(patient, images_dir, masks_dir)
        n_images += len(list(patient.glob("*.tif"))) // 2   # conta pares

    print(f"[{split_name:>5}] {len(patients):>3} pacientes | {n_images:>4} slices copiados")

print(f"\nDataset salvo em: {OUTPUT_DIR.resolve()}")
