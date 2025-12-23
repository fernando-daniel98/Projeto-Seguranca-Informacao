# Projeto Final: Segurança da Informação
Projeto final da disciplina de Segurança da Informação

## Descrição
Este projeto tem como objetivo fazer avaliações e análises comparativas entre três diferentes abordagens de segurança da informação, abordagens essas que serão feitas para um mesmo conjunto de dados. O objetivo é entender as vantagens e desvantagens de cada abordagem, bem como identificar qual delas é mais eficaz em diferentes cenários.

## 📁 Estrutura do Repositório

```
Projeto-Seguranca-Informacao/
├── database.py                 # Gera dataset NUAA (multi-seed, separação por sujeito)
├── database2.py                # Gera dataset CASIA-FASD
├── database_old.py             # Versão legada (baseline)
├── requirements.txt            # Dependências do projeto
├── dataset/                    # Dataset NUAA organizado (gerado por database.py)
├── dataset2/                   # Dataset CASIA-FASD organizado (gerado por database2.py)
│
├── Metodo_1/                   # LBP + SVM
│   ├── extract_lbp.py          # Extração de features LBP
│   ├── svm.py                  # Treinamento do classificador SVM
│   ├── metrics.py              # Avaliação e métricas
│   ├── visualize_lbp.py        # Visualização das features LBP
│   ├── exploratory_analises.py # Análise exploratória do dataset
│   ├── test_new_images.py      # Testar com novas imagens
│   ├── data/                   # Features extraídas (.npy)
│   ├── models/                 # Modelos treinados (.pkl)
│   ├── results/                # Resultados e gráficos
│   └── CASIA/                  # Treino cruzado CASIA→NUAA
│       ├── extract_lbp_casia.py
│       ├── train_svm_casia.py
│       └── test_nuaa.py
│
├── Metodo_2/                   # VGG16 Transfer Learning
│   ├── main.py                 # Treinamento do modelo VGG16
│   ├── metrics.py              # Avaliação e métricas
│   ├── run.sh                  # Script para executar pipeline completo
│   ├── models/                 # Modelos treinados (.keras)
│   └── results/                # Resultados e gráficos
│
└── Metodo_3/                   # Histogramas de Cor + SVM
    ├── 1_extract_cor.py        # Extração de histogramas HSV/YCbCr
    ├── 2_svm.py                # Treinamento do classificador SVM
    ├── 3_metricas.py           # Avaliação e métricas
    ├── data/                   # Features extraídas
    ├── models/                 # Modelos treinados
    └── results/                # Resultados e gráficos
```

---

## 🛠️ Como Replicar o Projeto

### Pré-requisitos

1. **Python 3.8+** instalado
2. **Conta no Kaggle** com API configurada (para download automático dos datasets)
3. **GPU (opcional)** para o Método 2 (VGG16)

### Passo 1: Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/Projeto-Seguranca-Informacao.git
cd Projeto-Seguranca-Informacao
```

### Passo 2: Criar Ambiente Virtual e Instalar Dependências

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Passo 3: Configurar Credenciais do Kaggle

```bash
# Baixe kaggle.json em: https://www.kaggle.com/settings
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### Passo 4: Gerar os Datasets

```bash
# Dataset NUAA (usado por Metodo_1 e Metodo_3)
python database.py

# Dataset CASIA-FASD (usado por Metodo_2)
python database2.py
```

---

## 📘 Método 1: LBP + SVM

Extrai features de textura usando Local Binary Patterns e classifica com SVM.

### Execução Completa

```bash
cd Metodo_1

# 1. Extrair features LBP (com separação por sujeito)
python extract_lbp.py --split-mode group-subject

# 2. Treinar SVM
python svm.py

# 3. Gerar métricas e visualizações
python metrics.py
```

### Opções Avançadas

```bash
# Validação cruzada por sujeito (GroupKFold)
python svm.py --subject-cv --k 5

# Visualizar features LBP
python visualize_lbp.py

# Análise exploratória do dataset
python exploratory_analises.py

# Testar com novas imagens
python test_new_images.py --dir /caminho/para/imagens
```

### Treino Cruzado CASIA → NUAA

```bash
cd Metodo_1/CASIA

# 1. Extrair features do CASIA
python extract_lbp_casia.py

# 2. Treinar SVM no CASIA
python train_svm_casia.py

# 3. Testar no NUAA
python test_nuaa.py
```

### Arquivos Gerados

- `data/X_train_lbp.npy`, `y_train_lbp.npy`, etc.
- `models/metodo1_lbp_svm.pkl`
- `results/results_metodo1.txt`
- `results/confusion_matrix_metodo1.png`
- `results/roc_curve_metodo1.png`

---

## 📗 Método 2: VGG16 Transfer Learning

Usa transfer learning com a rede VGG16 pré-treinada no ImageNet.

### Execução Completa (Script Automatizado)

```bash
cd Metodo_2
chmod +x run.sh
./run.sh
```

### Execução Manual

```bash
cd Projeto-Seguranca-Informacao

# 1. Treinar o modelo VGG16
python -m Metodo_2.main

# 2. Gerar métricas e visualizações
python -m Metodo_2.metrics
```

### Arquivos Gerados

- `Metodo_2/models/metodo2_vgg16_best.keras`
- `Metodo_2/results/results_metodo2.txt`
- `Metodo_2/results/confusion_matrix_metodo2.png`
- `Metodo_2/results/roc_curve_metodo2.png`
- `Metodo_2/results/score_distribution_metodo2.png`

---

## 📙 Método 3: Histogramas de Cor + SVM

Extrai histogramas nos espaços de cor HSV e YCbCr e classifica com SVM.

### Execução Completa

```bash
cd Metodo_3

# 1. Extrair características de cor
python 1_extract_cor.py

# 2. Treinar SVM
python 2_svm.py

# 3. Gerar métricas e visualizações
python 3_metricas.py
```

### Arquivos Gerados

- `data/X_train_cor.npy`, `y_train_cor.npy`, etc.
- `models/metodo3_cor_svm.pkl`
- `results/results_metodo3.txt`
- `results/confusion_matrix_metodo3.png`
- `results/roc_curve_metodo3.png`

---

## 📊 Parâmetros de Comparação

| Métrica | Descrição |
|---------|-----------|
| **Acurácia** | Porcentagem total de classificações corretas |
| **Precisão** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1-Score** | Média harmônica entre Precisão e Recall |
| **FAR** | False Acceptance Rate (fake aceito como real) |
| **FRR** | False Rejection Rate (real rejeitado como fake) |
| **HTER** | (FAR + FRR) / 2 |
| **EER** | Equal Error Rate (ponto onde FAR = FRR) |
| **AUC-ROC** | Área sob a curva ROC |

---

## 🔬 Metodologia de Avaliação

1. **Separação por Sujeitos:** Imagens do mesmo indivíduo não aparecem em train e test
2. **Balanceamento:** Undersampling da classe majoritária no treino
3. **Multi-Seed:** Execução com 5 seeds para robustez estatística
4. **Validação Cruzada:** GroupKFold por sujeito (opcional)

---

## 🗂️ Datasets Utilizados

| Dataset | Origem | Script |
|---------|--------|--------|
| **NUAA** | [Kaggle: aleksandrpikul222/nuaaaa](https://www.kaggle.com/datasets/aleksandrpikul222/nuaaaa) | `database.py` |
| **CASIA-FASD** | [Kaggle: minhnh2107/casiafasd](https://www.kaggle.com/datasets/minhnh2107/casiafasd) | `database2.py` |

---

## 🐛 Resolução de Problemas

| Problema | Solução |
|----------|---------|
| `FileNotFoundError` nos `.npy` | Execute os scripts na ordem correta |
| Erro no download do Kaggle | Configure `~/.kaggle/kaggle.json` |
| Memória insuficiente (Método 2) | Reduza `BATCH_SIZE` em `main.py` |
| Modelo não encontrado | Verifique se o treinamento foi concluído |

---

## 👥 Colaboradores

- [Fernando Daniel Marelino](https://github.com/fernando-daniel98/)
- [Ícaro Travain Darwich da Rocha](https://github.com/Itravain)
- [Marcos Aquino](https://github.com/Marcos-Aquin0)
- [Mateus Vespasiano de Castro](https://github.com/mateusvdcastro)