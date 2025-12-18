"""
database.py
Baixa o dataset NUAA (aleksandrpikul222/nuaaaa) via KaggleHub,
detecta automaticamente ClientRaw (real) e ImposterRaw (fake),
organiza em dataset/train, dataset/val, dataset/test
e gera um CSV com caminhos, split e label.
"""

import os
import random
import shutil
from pathlib import Path
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# CONFIGURAÇÕES
KAGGLE_DATASET = "aleksandrpikul222/nuaaaa"
OUTPUT_DIR = Path("dataset")
# Separação por IDs para evitar "Subject Leakage" 
SUBJECTS_VAL = ['0004', '0009'] 
SUBJECTS_TEST = ['0012', '0013', '0015'] 

random.seed(42)

# 1. DOWNLOAD
print("Baixando dataset...")
dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

# 2. LOCALIZAÇÃO DAS PASTAS
all_dirs = [p for p in dataset_path.rglob("*") if p.is_dir()]
client_dir = next((p for p in all_dirs if p.name.lower() == "clientraw"), None)
imposter_dir = next((p for p in all_dirs if p.name.lower() == "imposterraw"), None)

# 3. ESTRUTURA DE PASTAS
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

# 4. ORGANIZAÇÃO E BALANCEAMENTO
def collect_images_by_split(src_dir: Path):
    """Agrupa caminhos de imagens por split baseado no ID do sujeito."""
    data_map = {"train": [], "val": [], "test": []}
    subject_folders = [f for f in src_dir.iterdir() if f.is_dir()]
    
    for folder in subject_folders:
        s_id = folder.name
        split_name = "test" if s_id in SUBJECTS_TEST else ("val" if s_id in SUBJECTS_VAL else "train")
        data_map[split_name].extend(list(folder.glob("*.jpg")))
    return data_map

print("Processando e balanceando o Treino...")
reais_map = collect_images_by_split(client_dir)
fakes_map = collect_images_by_split(imposter_dir)

# Balanceamento apenas no TRAIN (Undersampling da classe majoritária)
min_train = min(len(reais_map["train"]), len(fakes_map["train"]))
random.shuffle(reais_map["train"])
random.shuffle(fakes_map["train"])
reais_map["train"] = reais_map["train"][:min_train]
fakes_map["train"] = fakes_map["train"][:min_train]

def copy_files(files_dict, label):
    for split, files in files_dict.items():
        for img in files:
            # Nome único: sujeito_nomeoriginal.jpg
            subject_id = img.parent.name
            dest = OUTPUT_DIR / split / label / f"{subject_id}_{img.name}"
            if not dest.exists():
                shutil.copy2(img, dest)

copy_files(reais_map, "real")
copy_files(fakes_map, "fake")

# 5. CSV E GRÁFICO (ORDEM CRESCENTE)
def get_folder_counts(base_dir='dataset'):
    data = []
    # Percorre as subpastas físicas
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(base_dir, split, label)
            if os.path.exists(path):
                # Conta arquivos reais no diretório
                count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                data.append({'split': split, 'label': label, 'Quantidade': count})
    return pd.DataFrame(data)

df_folders = get_folder_counts()

if not df_folders.empty:
    split_totals = df_folders.groupby('split')['Quantidade'].sum().sort_values().index.tolist()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df_folders, x='split', y='Quantidade', hue='label', 
                    order=split_totals, palette={'real': 'forestgreen', 'fake': 'firebrick'})

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{int(height)}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', fontweight='bold', xytext=(0, 5), textcoords='offset points')

    plt.title("Quantidade de Fotos", fontsize=14)
    plt.ylabel("Número de Arquivos")
    plt.xlabel("Conjunto")
    
    plt.savefig("contagem_real_pastas.png", dpi=300, bbox_inches='tight')
    
    print("\nCONTAGEM DE FOTOS:")
    print("-" * 30)
    print(df_folders.to_string(index=False))
    print("-" * 30)
    print(f"Ordem no gráfico (Crescente): {split_totals}")
else:
    print("Erro: A pasta 'dataset' não foi encontrada. Certifique-se de que o script 'database.py' foi executado com sucesso.")