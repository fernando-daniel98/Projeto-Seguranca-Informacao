import os
import random
import shutil
from pathlib import Path
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================
KAGGLE_DATASET = "aleksandrpikul222/nuaaaa"
# Definimos 5 seeds para testar a estabilidade do modelo
SEEDS = [42, 10, 23, 56, 89] 
SUBJECTS_VAL = ['0004', '0009'] 
SUBJECTS_TEST = ['0012', '0013', '0015'] 

# 1. DOWNLOAD (Ocorre apenas uma vez na pasta cache)
print("Baixando dataset original via KaggleHub...")
dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))

# 2. LOCALIZAÇÃO DAS PASTAS ORIGINAIS
all_dirs = [p for p in dataset_path.rglob("*") if p.is_dir()]
client_dir = next((p for p in all_dirs if p.name.lower() == "clientraw"), None)
imposter_dir = next((p for p in all_dirs if p.name.lower() == "imposterraw"), None)

def collect_images_by_split(src_dir: Path):
    """Mapeia imagens por split baseado no ID do sujeito."""
    data_map = {"train": [], "val": [], "test": []}
    subject_folders = [f for f in src_dir.iterdir() if f.is_dir()]
    for folder in subject_folders:
        s_id = folder.name
        split_name = "test" if s_id in SUBJECTS_TEST else ("val" if s_id in SUBJECTS_VAL else "train")
        data_map[split_name].extend(list(folder.glob("*.jpg")))
    return data_map

# Coleta inicial
reais_original = collect_images_by_split(client_dir)
fakes_original = collect_images_by_split(imposter_dir)

# =============================================================================
# 3. PROCESSAMENTO MULTI-SEED
# =============================================================================
for s in SEEDS:
    print(f"\nProcessando SEED: {s}")
    random.seed(s)
    current_out_dir = Path(f"dataset_seed_{s}")

    # Cria estrutura de pastas para esta seed específica
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            (current_out_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Balanceamento do TREINO para esta seed
    # O Undersampling garante que o SVM não tenha viés estatístico
    min_train = min(len(reais_original["train"]), len(fakes_original["train"]))
    
    # Criamos cópias para não alterar a lista original no próximo loop
    train_reais = list(reais_original["train"])
    train_fakes = list(fakes_original["train"])
    
    random.shuffle(train_reais)
    random.shuffle(train_fakes)
    
    # Seleção balanceada
    current_reais = {"train": train_reais[:min_train], "val": reais_original["val"], "test": reais_original["test"]}
    current_fakes = {"train": train_fakes[:min_train], "val": fakes_original["val"], "test": fakes_original["test"]}

    def copy_files(files_dict, label, out_dir):
        for split, files in files_dict.items():
            for img in files:
                subject_id = img.parent.name
                dest = out_dir / split / label / f"{subject_id}_{img.name}"
                if not dest.exists():
                    shutil.copy2(img, dest)

    copy_files(current_reais, "real", current_out_dir)
    copy_files(current_fakes, "fake", current_out_dir)

    # Gera CSV para esta seed
    rows = []
    for split in ["train", "val", "test"]:
        for label in ["real", "fake"]:
            for img in (current_out_dir / split / label).glob("*.jpg"):
                rows.append([str(img), split, label])
    pd.DataFrame(rows, columns=["filepath", "split", "label"]).to_csv(f"dataset_seed_{s}.csv", index=False)

# =============================================================================
# 4. GRÁFICO DE CONTAGEM (Exemplo da última seed processada)
# =============================================================================
def generate_report(base_dir):
    data = []
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(base_dir, split, label)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                data.append({'split': split, 'label': label, 'Quantidade': count})
    df = pd.DataFrame(data)
    
    # Ordem crescente de volume total
    split_totals = df.groupby('split')['Quantidade'].sum().sort_values().index.tolist()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(data=df, x='split', y='Quantidade', hue='label', 
                    order=split_totals, palette={'real': 'forestgreen', 'fake': 'firebrick'})

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontweight='bold', xytext=(0, 5), textcoords='offset points')

    plt.title(f"Distribuição de Fotos (Seed {s})", fontsize=14)
    plt.savefig(f"contagem_fotos.png", dpi=300, bbox_inches='tight')
    print(f"\nRelatório da Seed {s} gerado com sucesso.")

generate_report(f"dataset_seed_{SEEDS[-1]}")
print("\nProcesso Multi-Seed concluído!")