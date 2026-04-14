# Projeto Final — Segmentação de Tumor Cerebral (LGG MRI)

## Contexto geral

Este é um projeto final de um curso de deep learning. O objetivo é treinar um modelo de **segmentação semântica** para identificar regiões tumorais em imagens de ressonância magnética cerebral, comparar múltiplas arquiteturas e documentar tudo em um relatório técnico.

**Dataset:** [LGG MRI Segmentation — Kaggle](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)  
**Tarefa:** Segmentação binária pixel a pixel (tumor = 1, fundo = 0)  
**Prazo:** 14 dias a partir do início  

---

## O que já foi entendido e decidido

### Sobre a tarefa de segmentação
- Diferente de classificação (1 label por imagem) e detecção (bounding box), segmentação classifica **cada pixel** da imagem
- A saída do modelo é uma **máscara binária** com a mesma resolução da imagem de entrada
- O modelo recebe um tensor `(H × W × 3)` e produz `(H × W × 1)`
- A arquitetura precisa de estrutura **encoder → decoder** (ex: U-Net) para reconstruir a saída pixel a pixel

### Sobre o dataset
- **110 pacientes** do TCGA (The Cancer Genome Atlas)
- **3.929 pares** de imagem + máscara no total
- Resolução: **256 × 256 pixels**
- Formato dos arquivos: `.tif`

### Estrutura de pastas após descompactar

```
lgg-mri-segmentation/
├── kaggle_3m/
│   ├── TCGA_CS_4941_19960909/
│   │   ├── TCGA_CS_4941_19960909_1.tif        ← imagem (3 canais RGB)
│   │   ├── TCGA_CS_4941_19960909_1_mask.tif   ← máscara (binária)
│   │   ├── TCGA_CS_4941_19960909_2.tif
│   │   ├── TCGA_CS_4941_19960909_2_mask.tif
│   │   └── ... (20 a 88 fatias por paciente)
│   ├── TCGA_CS_4942_19970222/
│   ├── TCGA_CS_4943_20000902/
│   └── ... (110 pastas no total)
└── data.csv   ← metadados genômicos dos pacientes (não usado no treino)
```

**Lógica de nomeação:** cada imagem tem seu par de máscara identificado pelo sufixo `_mask`. O código deve encontrar todos os `.tif` que **não** terminam em `_mask` e montar os pares automaticamente.

### Por que cada paciente tem múltiplos arquivos
Um exame de MRI não é uma imagem única — o equipamento percorre o cérebro em fatias 2D (como fatiar um pão). Cada fatia é salva como um arquivo separado. A unidade de treino é sempre o **par fatia + máscara**, não o paciente inteiro.

### Desbalanceamento de classes
- **65% das máscaras são completamente pretas** (fatias sem tumor visível)
- **35% contêm tumor** (pixels brancos na máscara)
- Isso precisa ser tratado na escolha da função de perda e/ou na estratégia de amostragem

---

## Plano de execução (14 dias)

### Etapa 1 — Definição do problema ✅
- Dataset: LGG MRI Segmentation
- Tarefa: segmentação binária de tumor cerebral
- Desafios: desbalanceamento de classes, variabilidade anatômica entre pacientes

### Etapa 2 — Ambiente e dependências
**Stack definida:**
- Python 3.10
- PyTorch 2.x
- `segmentation-models-pytorch` — fornece U-Net, U-Net++, DeepLabV3+ prontos
- `albumentations` — augmentations
- `opencv-python` — leitura de imagens `.tif`
- `matplotlib` / `seaborn` — visualização
- `tqdm` — progress bars
- `scikit-learn` — split e métricas auxiliares

**Entregas desta etapa:**
- [ ] `requirements.txt` com todas as dependências e versões fixadas

### Etapa 3 — Preparação do dataset
**O que fazer:**
1. Percorrer todas as 110 pastas dentro de `kaggle_3m/`
2. Encontrar todos os `.tif` que não terminam em `_mask`
3. Montar lista de pares `(caminho_imagem, caminho_mascara)`
4. Fazer split **por paciente** (não por imagem!) em train/val/test — 70/15/15
5. Criar classe `Dataset` do PyTorch
6. Definir augmentations

**Atenção no split:** o split deve ser feito por paciente, não por imagem aleatoriamente. Se fatias do mesmo paciente caírem em train e val, o modelo aprende o paciente específico, não generaliza.

**Augmentations planejadas:**
- Flip horizontal ✅ — válido: o cérebro tem simetria aproximada entre hemisférios
 profundas), gerando imagens sem correspondência clínica real
- Rotação leve (±15°) ✅ — válido: pacientes não ficam perfeitamente alinhados no scanner
- Variação de brilho e contraste ✅ — válido: equipamentos e protocolos diferentes geram intensidades distintas
- Elastic transform leve ✅ — válido: simula variação anatômica entre pacientes
- Zoom/crop leve ✅ — válido: simula diferentes posicionamentos no scanner
- Normalização (média e desvio padrão do ImageNet ou calculado no dataset)

**Regra geral para augmentations em imagem médica:** só aplicar transformações que gerem imagens anatomicamente plausíveis. Cada augmentation deve passar pelo teste: "isso poderia ser um exame real?"

**Entregas desta etapa:**
- [ ] Seção no relatório explicando as escolhas
- [ ] Classe `BrainMRIDataset` no notebook

### Etapa 4 — Modelos e treinamento
**Arquiteturas a comparar (mínimo 2):**

| Modelo | Backbone | Justificativa |
|--------|----------|---------------|
| U-Net | ResNet34 | Baseline clássico para segmentação médica |
| U-Net++ | EfficientNet-B4 | Arquitetura mais robusta, conexões densas |
| DeepLabV3+ | ResNet50 | Opcional — terceiro modelo para comparação |

**Configurações de treino:**
- Função de perda: `DiceLoss` ou combinação `Dice + BCE` (trata desbalanceamento)
- Otimizador: `Adam` com `lr=1e-4`
- Scheduler: `ReduceLROnPlateau`
- Épocas: 50–100 (com early stopping)
- Batch size: 16 (ajustar conforme VRAM)
- Métricas: **Dice Coefficient** (principal) + **IoU**

**Entregas desta etapa:**
- [ ] Notebook completo com treinamento
- [ ] Melhor checkpoint salvo (`.pth`)
- [ ] Summary do modelo (`torchsummary` ou `torchinfo`)
- [ ] Curvas de loss e métrica por época (gráfico)

### Etapa 5 — Análise de resultados
**O que produzir:**
- Tabela comparando todos os modelos (Dice, IoU, F1)
- Visualizações: imagem original | máscara real | máscara predita (lado a lado)
- Exemplos de **bons resultados** e de **falhas** (análise qualitativa)
- Discussão crítica: o que funcionou, o que não funcionou e por quê

**Entregas desta etapa:**
- [ ] Seção de resultados no relatório com tabela comparativa

### Etapa 6 — Repositório e documentação
**Estrutura de repositório sugerida:**
```
/
├── data/                  ← não versionar (adicionar ao .gitignore)
├── notebooks/
│   └── train.ipynb
├── src/
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── checkpoints/           ← não versionar
├── requirements.txt
├── README.md
└── PROJECT_CONTEXT.md     ← este arquivo
```

**README deve conter:**
- Descrição do problema
- Tecnologias utilizadas
- Como configurar o ambiente (`pip install -r requirements.txt`)
- Como executar o treinamento
- Resultados obtidos (tabela + imagens de exemplo)

**Pitch (3 minutos máximo):**
1. Abertura e contextualização (o que é LGG, por que segmentar)
2. Descrição do problema (dataset, desafios)
3. Solução (pipeline, arquiteturas testadas)
4. Diferenciais (o que foi além do básico)
5. Resultados (Dice, IoU, visualizações)

---

## Próximos passos imediatos

1. Criar `requirements.txt`
2. Escrever a classe `BrainMRIDataset` que monta os pares imagem/máscara
3. Fazer o split por paciente e verificar a distribuição de classes em cada split
4. Implementar o loop de treino com U-Net + ResNet34 como baseline

---

## Referências e decisões técnicas pendentes

- [ ] Decidir se vai usar pesos pré-treinados no ImageNet para os backbones (transfer learning)
- [ ] Definir threshold de binarização na saída do modelo (padrão: 0.5)
- [ ] Decidir se vai filtrar fatias completamente negativas do treino ou manter todas
- [ ] Escolher se roda local ou Google Colab (impacta batch size e épocas viáveis)
