# Projeto Final: Segurança da Informação
Projeto final da disciplina de Segurança da Informação

## Descrição
Este projeto tem como objetivo fazer avaliações e análises comparativas entre três diferentes abordagens de segurança da informação, abordagens essas que serão feitas para um mesmo conjunto de dados. O objetivo é entender as vantagens e desvantagens de cada abordagem, bem como identificar qual delas é mais eficaz em diferentes cenários.

## Parâmetros utilizados para comparação
- Acurácia
- Precisão 
- Recall

Matriz de confusão


## Colaboradores
- [Fernando Daniel Marelino](https://github.com/fernando-daniel98/)
- [Ícaro Travain Darwich da Rocha](https://github.com/Itravain)
- [Marcos Aquino](https://github.com/Marcos-Aquin0)
- [Mateus Vespasiano de Castro](https://github.com/mateusvdcastro)

## Instruções de uso

Requisitos
- Python 3.8+
- Instale dependências usando o arquivo requirements.txt:
  ```
  pip install -r requirements.txt
  ```
  (coloque-se na raiz do repositório antes de executar o comando)

Passos para executar o pipeline
1. Baixar e organizar o dataset (script usa kagglehub):
   ```
   python database.py
   ```
   Saída: pasta `dataset/` com subpastas train/val/test e `dataset.csv`.

2. Extrair características (Método 3 — cor):
   ```
   python Metodo_3/1_extract_cor.py
   ```
   Saída: arquivos em `./Metodo_3/data/`:
   - X_train_color.npy, y_train_color.npy, X_test_color.npy, y_test_color.npy

3. Treinar SVM:
   ```
   python Metodo_3/2_svm.py
   ```
   Saída: modelo salvo em `./Metodo_3/models/metodo3_svm_color.pkl`

4. Avaliar e gerar métricas/plots:
   ```
   python Metodo_3/3_metricas.py
   ```
   Saída: `./Metodo_3/results/results_metodo3.txt` e imagens em `./Metodo_3/results/`

Onde estão os scripts / onde colocar código
- Arquivos principais:
  - `database.py` (na raiz) — download e organização do dataset
  - `Metodo_3/1_extract_cor.py` — extração de características (cor)
  - `Metodo_3/2_svm.py` — treinamento do SVM
  - `Metodo_3/3_metricas.py` — avaliação e geração de plots
- Estrutura esperada (resumida):
  ```
  .
  ├─ database.py
  ├─ requirements.txt
  ├─ Metodo_1/
  ├─ Metodo_2/
  └─ Metodo_3/
     ├─ 1_extract_cor.py
     ├─ 2_svm.py
     └─ 3_metricas.py
  ```

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