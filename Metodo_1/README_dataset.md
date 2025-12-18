# Método 1 - Uso do Dataset Separado por Sujeito

## Alterações Realizadas

O Método 1 foi atualizado para usar o dataset gerado por `database.py` (na raiz do projeto), que implementa separação adequada por sujeito para evitar **subject leakage**.

### O que mudou:

1. **extract_lbp.py**:
   - Agora aponta para `../dataset/` (raiz do projeto) em vez de `./dataset/` (pasta local)
   - Parser de nomes de arquivo atualizado para suportar ambos os formatos:
     - Original: `0011_01_07_03_202.jpg`
     - Database.py: `0011_0011_01_07_03_202.jpg` (sujeito duplicado)

2. **exploratory_analises.py**:
   - Dataset padrão alterado para `../dataset/` (raiz do projeto)
   - Mantém compatibilidade com argumento `--dataset-dir` para especificar outro caminho

3. **Separação por Sujeito no database.py**:
   - **VAL**: Sujeitos 0004, 0009
   - **TEST**: Sujeitos 0012, 0013, 0015
   - **TRAIN**: Demais sujeitos (com balanceamento por undersampling)

## Como Usar

### 1. Gerar o Dataset (se ainda não foi feito)

Na raiz do projeto, execute:

```bash
python database.py
```

Isso irá:
- Baixar o dataset NUAA via KaggleHub
- Organizar em `dataset/train`, `dataset/val`, `dataset/test`
- Cada pasta contém subpastas `real/` e `fake/`
- Balancear apenas o conjunto de treino
- Gerar gráfico de contagem

### 2. Extrair Features LBP

Entre na pasta do Método 1 e extraia as features:

```bash
cd Metodo_1
python extract_lbp.py --split-mode filesystem
```

Parâmetros importantes:
- `--split-mode filesystem`: Usa a estrutura de pastas já criada pelo database.py (recomendado)
- `--split-mode group-subject`: Recria splits separando por sujeito (alternativa)
- `--split-mode official-like`: Usa tamanhos da literatura + validação por sujeito

### 3. Análise Exploratória (Opcional)

```bash
python exploratory_analises.py
```

Gera um relatório em `results/exploratory_dataset_summary.txt` com:
- Contagem de imagens por split e classe
- Distribuição de extensões
- Estatísticas de tamanho e intensidade

### 4. Treinar SVM

```bash
python svm.py
```

Ou com validação cruzada por sujeito (GroupKFold):

```bash
python svm.py --subject-cv --k 5
```

### 5. Avaliar Métricas

```bash
python metrics.py
```

Gera `results/results_metodo1.txt` com:
- Acurácia, Precisão, Recall
- FAR, FRR, HTER, EER
- AUC-ROC
- Matriz de confusão

### 6. Testar com Novas Imagens

```bash
python test_new_images.py --dir /caminho/para/pasta
```

Modos:
- **Validação**: Se a pasta tiver subpastas `real/` e `fake/`, calcula métricas
- **Inferência**: Se tiver imagens soltas, mostra predições individuais

## Vantagens da Separação por Sujeito

✅ **Evita subject leakage**: Imagens do mesmo sujeito não aparecem em train e test  
✅ **Generalização real**: Testa a capacidade do modelo em sujeitos inéditos  
✅ **Reprodutibilidade**: Seeds fixos garantem splits consistentes  
✅ **Compatível com literatura**: Pode simular divisões reportadas em papers  

## Estrutura de Pastas Esperada

```
Projeto-Seguranca-Informacao/
├── database.py                    # Gera dataset na raiz
├── dataset/                       # Criado por database.py
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   ├── val/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
└── Metodo_1/
    ├── extract_lbp.py             # Aponta para ../dataset
    ├── exploratory_analises.py    # Aponta para ../dataset
    ├── svm.py
    ├── metrics.py
    ├── test_new_images.py
    └── data/                      # Features extraídas
        ├── X_train_lbp.npy
        ├── y_train_lbp.npy
        └── ...
```

## Notas Importantes

- O `database.py` faz **balanceamento apenas no treino** (undersampling da classe majoritária)
- Val e Test **não são balanceados**, refletindo a distribuição natural
- Para usar o dataset antigo (local), especifique: `python extract_lbp.py --split-mode filesystem` e garanta que `Metodo_1/dataset/` existe
- Sempre execute `database.py` primeiro, na raiz do projeto
