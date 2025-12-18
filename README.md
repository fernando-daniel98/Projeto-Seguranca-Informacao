# Projeto Final: Segurança da Informação
Projeto final da disciplina de Segurança da Informação

## Descrição
Este projeto tem como objetivo fazer avaliações e análises comparativas entre três diferentes abordagens de segurança da informação, abordagens essas que serão feitas para um mesmo conjunto de dados. O objetivo é entender as vantagens e desvantagens de cada abordagem, bem como identificar qual delas é mais eficaz em diferentes cenários.

## Parâmetros utilizados para comparação
- Acurácia
- Precisão 
- Recall

- AUC (ROC)

Matriz de confusão


## Colaboradores
- [Fernando Daniel Marelino](https://github.com/fernando-daniel98/)
- [Ícaro Travain Darwich da Rocha](https://github.com/Itravain)
- [Marcos Aquino](https://github.com/Marcos-Aquin0)
- [Mateus Vespasiano de Castro](https://github.com/mateusvdcastro)

## 📌 Descrição do Projeto
Este projeto realiza uma análise comparativa entre três abordagens de segurança aplicadas à detecção de ataques de apresentação (fotos impressas) em sistemas biométricos. O objetivo principal é avaliar a robustez forense de cada método, garantindo que o sistema aprenda características reais de vivacidade em vez de apenas memorizar rostos.

## 🔬 Metodologia de Avaliação Rigorosa
Diferente de abordagens simplistas, esta versão implementa três pilares críticos para segurança biométrica:

1.  **Separação por Indivíduos (Subject-Independent):** A divisão entre treino e teste é baseada em IDs de sujeitos exclusivos. Isso garante que o modelo seja testado em rostos que ele **nunca viu** durante o treinamento, eliminando o vazamento de dados (*Subject Leakage*).
2.  **Treino Balanceado (Undersampling):** A quantidade de imagens reais e falsas no conjunto de treino é igualada para remover qualquer viés estatístico do classificador SVM.
3.  **Análise Multi-Seed:** O pipeline é executado sob 5 sementes de aleatoriedade ($SEEDS = [42, 10, 23, 56, 89]$) para gerar médias e desvios padrões robustos, garantindo a reprodutibilidade dos resultados.



## 📊 Parâmetros de Comparação
As métricas utilizadas avaliam o desempenho sob diferentes perspectivas de segurança:
* **Acurácia:** Porcentagem total de classificações corretas.
* **HTER (Half Total Error Rate):** Média entre as taxas de falsa aceitação (FAR) e falsa rejeição (FRR). $$HTER = \frac{FAR + FRR}{2}$$
* **EER (Equal Error Rate):** O ponto de equilíbrio onde $FAR = FRR$, indicando a precisão intrínseca do classificador.
* **Matriz de Confusão, Curva ROC e Distribuição de Scores.**

---

## 🚀 Instruções de Uso

### 1. Preparação do Dataset
Existem duas versões do script de organização de dados para fins de comparação:

* **Versão Atualizada (`database.py`):** Aplica a separação por IDs de sujeitos, balanceamento de classes e cria a estrutura para as 5 sementes.
    ```bash
    python database.py
    ```
    *Saída:* Cria pastas `dataset_seed_X/` e arquivos `dataset_seed_X.csv`.

* **Versão Legada (`database_old.py`):** Mantém a organização original sem separação de IDs ou balanceamento. Serve apenas como baseline para observar o impacto do vazamento de dados nos resultados.

### 2. Execução do Método 3 (Cores + SVM)
O pipeline deve ser executado para as 5 sementes para gerar o resumo estatístico final:

1.  **Extração de Características:**
    ```bash
    python Metodo_3/1_extract_cor.py
    ```
    *Extrai histogramas nos espaços de cor HSV e YCbCr para cada semente.*

2.  **Treinamento do Modelo:**
    ```bash
    python Metodo_3/2_svm.py
    ```
    *Treina 5 modelos SVM (RBF) independentes.*

3.  **Avaliação e Métricas:**
    ```bash
    python Metodo_3/3_metricas.py
    ```
    *Gera gráficos e arquivos `.txt` individuais por semente e exibe o resumo estatístico final ($\mu \pm \sigma$).*

---

## 📁 Estrutura do Repositório
```text
.
├── database.py             # Versão atualizada (Subject ID + Balanceamento + Multi-seed)
├── database_old.py         # Versão original para comparação (Baseline)
├── requirements.txt        # Dependências (OpenCV, Scikit-learn, etc.)
├── Metodo_1/
├── Metodo_2/
├── Metodo_3/
│   ├── 1_extract_cor.py    # Extração de histogramas HSV/YCbCr
│   ├── 2_svm.py            # Treinamento do classificador SVM
│   └── 3_metricas.py       # Avaliação, plots e resumo estatístico
└── dataset/                # Gerado pelo database.py (Ignorado no Git)


Dicas e resolução de problemas
- Se ocorrer FileNotFoundError ao carregar .npy ou .pkl, verifique a ordem: database.py → Metodo_3/1_extract_cor.py → Metodo_3/2_svm.py → Metodo_3/3_metricas.py.
- Se o download via Kaggle falhar, configure as credenciais do Kaggle ou faça download manual.
- Use um ambiente virtual (venv/conda) para isolar dependências.

Contribuições
- Abra um issue ou pull request com melhorias no pipeline ou novas métricas/visualizações.

## Sobre o Dataset

- Nome / origem
  - Dataset utilizado: NUAA Face Anti-Spoofing (via Kaggle, referência usada: aleksandrpikul222/nuaaaa).
  - O script `database.py` baixa e organiza automaticamente esse dataset (requer KaggleHub e credenciais configuradas, ou download manual).

- Estrutura e rótulos
  - O dataset original apresenta duas pastas principais: ClientRaw (imagens REAIS) e ImposterRaw (imagens FALSAS).
  - Após executar `python database.py` a organização é:
    - dataset/train/real
    - dataset/train/fake
    - dataset/val/real
    - dataset/val/fake
    - dataset/test/real
    - dataset/test/fake
  - Nos scripts, usamos os rótulos:
    - 'real' → 1
    - 'fake' → 0

- Formato e pré-processamento
  - Arquivos esperados: imagens .jpg/.jpeg/.png/.bmp.
  - Os scripts de extração redimensionam para 224x224 (veja `Metodo_3/1_extract_cor.py`).
  - Ajuste de bins, normalização ou transformação podem ser alterados no script de extração.

- Como inspecionar o dataset e contar imagens
  - Via linha de comando:
    ```
    head -n 5 dataset.csv
    ```
  - Via Python (rápido check):
    ```
    python - <<'PY'
    import pandas as pd
    df = pd.read_csv('dataset.csv')
    print("Distribuição por label:")
    print(df['label'].value_counts())
    print("\nDistribuição por split:")
    print(df['split'].value_counts())
    PY
    ```
  - Ou com pandas interativo:
    ```
    import pandas as pd
    df = pd.read_csv('dataset.csv')
    display(df.head())
    ```

- Alterar proporção de splits
  - Para modificar os percentuais de train/val/test edite a constante SPLIT em `database.py`:
    ```
    SPLIT = (0.7, 0.2, 0.1)  # train, val, test
    ```

- Observações
  - Caso o download do Kaggle falhe, baixe manualmente e coloque as pastas ClientRaw/ImposterRaw dentro de uma pasta local, ou ajuste o caminho em `database.py`.
  - Verifique o CSV `dataset.csv` gerado pelo script para confirmar paths absolutos/relativos usados pelos scripts subsequentes.

## Nota sobre particionamento (evitar vazamento)

Para o dataset NUAA, é importante evitar que imagens do mesmo sujeito/sessão apareçam em treino/val/test, pois isso pode inflar muito os resultados.

- O [Metodo_1/extract_lbp.py](Metodo_1/extract_lbp.py) suporta refazer os splits a partir de toda a base com `--split-mode group-subject` (recomendado) ou `--split-mode group-session`.
- O [Metodo_1/svm.py](Metodo_1/svm.py) suporta validação cruzada por sujeito via `--subject-cv` (GroupKFold).