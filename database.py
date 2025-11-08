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

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

KAGGLE_DATASET = "aleksandrpikul222/nuaaaa"
OUTPUT_DIR = Path("dataset")          # pasta final organizada
SPLIT = (0.7, 0.2, 0.1)               # train, val, test
random.seed(42)                       # reprodutibilidade

# =============================================================================
# 1. DOWNLOAD DO DATASET
# =============================================================================

print("📥 Baixando dataset NUAA via KaggleHub...")
dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
dataset_path = Path(dataset_path)

print(f"Dataset baixado em: {dataset_path}")

# =============================================================================
# 2. MOSTRAR PARTE DA ESTRUTURA (DEBUG AMIGÁVEL)
# =============================================================================

print("\nEstrutura detectada (parcial):\n")
max_print_dirs = 30
count = 0
for root, dirs, files in os.walk(dataset_path):
    if count >= max_print_dirs:
        print("    ...")
        break
    level = root.replace(str(dataset_path), "").count(os.sep)
    indent = " " * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files[:3]:
        print(f"{indent}    {f}")
    count += 1

print("\n────────────────────────────────────────────\n")

# =============================================================================
# 3. Separar ClientRaw (REAL) e ImposterRaw (FAKE)
# =============================================================================

print("Procurando pastas 'ClientRaw' (real) e 'ImposterRaw' (fake)...")

all_dirs = [p for p in dataset_path.rglob("*") if p.is_dir()]

client_dir = next((p for p in all_dirs if p.name.lower() == "clientraw"), None)
imposter_dir = next((p for p in all_dirs if p.name.lower() == "imposterraw"), None)

if not client_dir or not imposter_dir:
    raise RuntimeError(
        "\nERRO: Não encontrei as pastas 'ClientRaw' e 'ImposterRaw'.\n"
        "Confira a estrutura impressa acima e ajuste os caminhos manualmente se necessário."
    )

print(f" Pasta de IMAGENS REAIS encontrada em: {client_dir}")
print(f" Pasta de IMAGENS FALSAS encontrada em: {imposter_dir}")

# =============================================================================
# 4. CRIA ESTRUTURA train / val / test
# =============================================================================

print("\nCriando estrutura dataset/train, val, test...")

for split_name in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (OUTPUT_DIR / split_name / label).mkdir(parents=True, exist_ok=True)

# =============================================================================
# 5. FUNÇÃO PARA SPLIT + CÓPIA
# =============================================================================

def split_and_copy(src_dir: Path, label: str):
    """Copia TODAS as imagens .jpg recursivamente para train/val/test."""
    images = [img for img in src_dir.rglob("*.jpg")]
    if len(images) == 0:
        raise RuntimeError(f"Nenhuma imagem .jpg encontrada em {src_dir}")

    random.shuffle(images)

    n = len(images)
    n_train = int(n * SPLIT[0])
    n_val = int(n * SPLIT[1])
    n_test = n - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    print(f"📸 {label.upper()}: {n} imagens -> "
          f"{len(splits['train'])} train, {len(splits['val'])} val, {len(splits['test'])} test")

    for split_name, file_list in splits.items():
        for img in file_list:
            dest = OUTPUT_DIR / split_name / label / img.name
            # se rodar mais de uma vez, evita erro de arquivo já existente
            if not dest.exists():
                shutil.copy2(img, dest)


# Real (ClientRaw) = label "real"
split_and_copy(client_dir, "real")

# Fake (ImposterRaw) = label "fake"
split_and_copy(imposter_dir, "fake")

# =============================================================================
# GERAR dataset.csv
# =============================================================================

print("\nGerando dataset.csv...")

rows = []
for split_name in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        folder = OUTPUT_DIR / split_name / label
        for img in folder.glob("*.jpg"):
            rows.append([str(img), split_name, label])

if not rows:
    raise RuntimeError("Nenhuma imagem foi copiada para a pasta dataset/. Verifique o script.")

df = pd.DataFrame(rows, columns=["filepath", "split", "label"])
df.to_csv("dataset.csv", index=False)

print("\nProcesso concluído com sucesso!")
print(f"Dataset organizado em: {OUTPUT_DIR.resolve()}")
print("CSV criado: dataset.csv")
print("Dados prontos para utilizar.\n")
