# README.md - CASIA to NUAA Experiment

## Descrição

Este experimento treina o Método 1 (LBP + SVM) no dataset **CASIA-FASD** (dataset2) e testa no dataset **NUAA** (dataset).

## Estrutura de Arquivos

```
CASIA/
├── extract_lbp_casia.py       # Extrai features LBP do CASIA
├── train_svm_casia.py          # Treina SVM com CASIA
├── test_nuaa.py                # Testa modelo treinado no NUAA
├── run_casia_to_nuaa.sh        # Script para executar todo o pipeline
├── data_casia/                 # Features LBP extraídas (gerado)
│   ├── X_train_lbp.npy
│   ├── y_train_lbp.npy
│   ├── X_val_lbp.npy
│   ├── y_val_lbp.npy
│   ├── X_test_lbp.npy
│   └── y_test_lbp.npy
├── models_casia/               # Modelos treinados (gerado)
│   ├── svm_casia_trained.pkl
│   └── best_params.txt
└── results_casia/              # Resultados do teste (gerado)
    ├── results_casia_to_nuaa.txt
    ├── confusion_matrix_casia_to_nuaa.png
    └── roc_curve_casia_to_nuaa.png
```

## Pré-requisitos

- Python 3.8+
- Dependências do projeto instaladas (ver `requirements.txt` na raiz)
- Dataset CASIA organizado em `dataset2/` (raiz do projeto)
- Dataset NUAA organizado em `dataset/` (raiz do projeto)

## Como Executar

### Opção 1: Script Automatizado (Recomendado)

```bash
cd Metodo_1/CASIA
bash run_casia_to_nuaa.sh
```

### Opção 2: Passo a Passo

```bash
cd Metodo_1/CASIA

# 1. Extrair features LBP do CASIA
python extract_lbp_casia.py

# 2. Treinar SVM com CASIA
python train_svm_casia.py

# 3. Testar no NUAA
python test_nuaa.py
```

## Fluxo do Experimento

1. **Extração de Features (extract_lbp_casia.py)**
   - Lê imagens do dataset CASIA (dataset2/train, val, test)
   - Redimensiona para 64x64
   - Converte para escala de cinza
   - Extrai histograma LBP (uniform, raio=1, pontos=8)
   - Salva features em `data_casia/`

2. **Treinamento (train_svm_casia.py)**
   - Carrega features LBP do CASIA
   - Aplica StandardScaler + SVM (kernel RBF)
   - GridSearchCV para otimizar C e gamma
   - Valida com conjunto de validação do CASIA
   - Retreina com train+val nos melhores parâmetros
   - Salva modelo em `models_casia/svm_casia_trained.pkl`

3. **Teste Cross-Dataset (test_nuaa.py)**
   - Carrega modelo treinado no CASIA
   - Extrai features LBP do NUAA (dataset/test)
   - Realiza predições
   - Calcula métricas: Acurácia, Precisão, Recall, F1, AUC, HTER
   - Gera visualizações (matriz de confusão, curva ROC)
   - Salva resultados em `results_casia/`

## Resultados Esperados

Os resultados do teste cross-dataset (CASIA→NUAA) serão salvos em:

- **results_casia_to_nuaa.txt**: Métricas de desempenho completas
- **confusion_matrix_casia_to_nuaa.png**: Visualização da matriz de confusão
- **roc_curve_casia_to_nuaa.png**: Curva ROC com AUC

## Notas

- O treinamento usa **class_weight="balanced"** para lidar com desbalanceamento de classes
- O GridSearchCV otimiza para **AUC-ROC** 
- Features LBP são extraídas de forma idêntica para CASIA e NUAA garantindo compatibilidade
- Este é um experimento de **generalização cross-dataset**, avaliando a capacidade do modelo de generalizar entre bases diferentes

## Troubleshooting

### Erro: "Dataset CASIA-FASD não encontrado"
Certifique-se de que `dataset2/` existe na raiz do projeto com a estrutura:
```
dataset2/
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

### Erro: "Dataset NUAA não encontrado"
Certifique-se de que `dataset/` existe na raiz do projeto com a mesma estrutura.

### Erro de memória
- Reduza o tamanho do dataset ou use uma máquina com mais RAM
- O GridSearchCV pode consumir muita memória; ajuste `n_jobs` se necessário

## Autor

Projeto de Segurança da Informação - Método 1 (LBP + SVM)
