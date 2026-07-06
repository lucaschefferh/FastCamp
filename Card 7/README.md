# Brain MRI Segmentation — LGG Tumor

Segmentação semântica de tumores cerebrais (Low-Grade Glioma) em imagens de MRI, usando PyTorch e `segmentation-models-pytorch`. O projeto cobre o pipeline completo: divisão do dataset, treinamento e inferência com avaliação quantitativa e visual.

## Dataset

[LGG Segmentation Dataset (Kaggle)](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) — 110 pacientes do TCGA, imagens FLAIR em `.tif` com máscaras binárias de segmentação.

```
archive/kaggle_3m/
    TCGA_<instituição>_<paciente>_<slice>.tif
    TCGA_<instituição>_<paciente>_<slice>_mask.tif
    data.csv
```

## Estrutura do Projeto

```
.
├── src/
│   ├── split_dataset.py   # divide o dataset por paciente (70/15/15 %)
│   ├── dataset.py         # BrainMRIDataset, transforms, dataloaders
│   ├── train.py           # loop de treinamento com early stopping
│   └── inference.py       # inferência no conjunto de teste + métricas + visualizações
├── archive/kaggle_3m/     # dataset original (não versionado)
├── dataset/               # splits gerados por split_dataset.py (não versionado)
│   ├── train/images+masks
│   ├── val/images+masks
│   └── test/images+masks
├── checkpoints/           # melhores pesos e histórico CSV por modelo
├── results/               # métricas e grades visuais por modelo
└── requirements.txt
```

## Instalação

```bash
pip install -r requirements.txt
```

> **Nota:** as versões no `requirements.txt` foram especificadas manualmente e podem não existir. Se `pip install` falhar, instale sem versões fixas ou gere um `requirements.txt` limpo com `pip freeze` após instalar manualmente.

## Como Usar

### 1. Dividir o dataset

```bash
python src/split_dataset.py
```

Divide por **paciente** (evita data leakage) em 70 % treino / 15 % validação / 15 % teste. Gera a pasta `dataset/` com subpastas `train/`, `val/` e `test/`.

### 2. Treinar

```bash
# Configuração padrão: UNet++ + EfficientNet-B4, 100 épocas
python src/train.py

# Personalizado
python src/train.py --model unet --backbone resnet34 --epochs 50 --batch_size 8
```

| Argumento        | Padrão             | Opções                             |
| ---------------- | ------------------- | ------------------------------------ |
| `--model`      | `unetpp`          | `unet`, `unetpp`, `deeplabv3p` |
| `--backbone`   | `efficientnet-b4` | qualquer encoder do `timm`         |
| `--epochs`     | `100`             | —                                   |
| `--batch_size` | `16`              | —                                   |
| `--lr`         | `1e-4`            | —                                   |
| `--patience`   | `15`              | early stopping                       |

O melhor checkpoint é salvo em `checkpoints/<modelo>_<backbone>_best.pth` e o histórico de métricas em `checkpoints/<modelo>_<backbone>_history.csv`.

### 3. Inferência e avaliação

```bash
# Seleciona automaticamente o melhor checkpoint disponível
python src/inference.py

# Checkpoint específico
python src/inference.py --checkpoint checkpoints/unet_resnet34_best.pth
```

Gera em `results/<modelo>_<backbone>/`:

- `test_predictions.csv` — métricas por amostra (Dice, IoU, Precision, Recall)
- `test_summary.csv` — resumo estatístico (total, com tumor, sem tumor)
- `best_predictions.png` — grade com as melhores segmentações
- `worst_predictions.png` — grade com as piores segmentações
- `metrics_distribution.png` — histogramas de Dice e IoU

## Resultados

Modelos treinados com early stopping (sem atingir as 100 épocas):

| Modelo | Backbone        | Melhor época | Val Dice |
| ------ | --------------- | ------------- | -------- |
| UNet   | ResNet34        | 8             | 0.585    |
| UNet++ | EfficientNet-B0 | 24            | 0.592    |

**Métricas no conjunto de teste (fatias com tumor):**

| Modelo             | Dice            | IoU             | Recall          | Detection Acc |
| ------------------ | --------------- | --------------- | --------------- | ------------- |
| UNet / ResNet34    | **0.731** | **0.644** | **0.788** | 91.9 %        |
| UNet++ / EffNet-B0 | 0.698           | 0.623           | 0.694           | 92.5 %        |

> O Dice "geral" (~0.27–0.29) é enganoso: inclui ~359 fatias sem tumor onde Dice=0 por definição matemática. O número relevante é o Dice nas **fatias com tumor** (acima).

## Arquitetura e Treinamento

- **Modelos:** U-Net, U-Net++ e DeepLabV3+ via `segmentation-models-pytorch`
- **Encoders:** pré-treinados no ImageNet
- **Loss:** 0.5 × DiceLoss + 0.5 × BCEWithLogitsLoss
- **Otimizador:** Adam com `ReduceLROnPlateau` (modo max, fator 0.5, patience 5)
- **Augmentations (treino):** flip horizontal, rotação ±15°, brilho/contraste, transformação elástica, crop aleatório
- **Métricas:** Dice coefficient e IoU (calculados após sigmoid + binarização em 0.5)
