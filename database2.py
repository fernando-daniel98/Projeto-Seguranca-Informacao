"""
database2.py
Baixa o dataset CASIA-FASD (minhnh2107/casiafasd) via KaggleHub,
organiza em dataset2/train, dataset2/val, dataset2/test
Usa apenas imagens COLOR (ignora depth).
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
KAGGLE_DATASET = "minhnh2107/casiafasd"
OUTPUT_DIR = Path("dataset2")

# Proporção para criar validação a partir do treino
VAL_RATIO = 0.15

random.seed(42)

# 1. DOWNLOAD
print("=" * 50)
print("Baixando dataset CASIA-FASD...")
print("=" * 50)
dataset_path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
print(f"Dataset baixado em: {dataset_path}")

# 2. LIMPAR DATASET2 EXISTENTE (se houver)
if OUTPUT_DIR.exists():
    print(f"\nRemovendo dataset2 existente...")
    shutil.rmtree(OUTPUT_DIR)

# 3. CRIAR ESTRUTURA DE SAÍDA
for split in ["train", "val", "test"]:
    for label in ["real", "fake"]:
        (OUTPUT_DIR / split / label).mkdir(parents=True, exist_ok=True)

# 4. COLETAR APENAS IMAGENS COLOR
print("\n" + "=" * 50)
print("COLETANDO IMAGENS (apenas COLOR)")
print("=" * 50)

IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp')

def classify_casia_image(img_path: Path) -> str:
    """
    Classifica uma imagem do CASIA-FASD pelo nome do arquivo.
    O padrão é: SUBJECT_VIDEO.avi_FRAME_LABEL.jpg
    Onde LABEL é 'real' ou 'fake'.
    """
    fname = img_path.name.lower()
    
    # Verificar se tem _real ou _fake no nome
    if '_real.' in fname or '_real' in fname.split('.')[0].split('_')[-1:]:
        return 'real'
    if '_fake.' in fname or '_fake' in fname.split('.')[0].split('_')[-1:]:
        return 'fake'
    
    # Fallback: verificar padrões HR
    # HR_1, HR_2 = real; HR_3, HR_4 = fake
    if 'hr_1' in fname or 'hr_2' in fname:
        # Mas se termina com _fake, é fake
        if '_fake' in fname:
            return 'fake'
        return 'real'
    if 'hr_3' in fname or 'hr_4' in fname:
        return 'fake'
    
    return 'unknown'

def determine_split(img_path: Path) -> str:
    """Determina se a imagem é de train ou test."""
    path_str = str(img_path).lower()
    
    if 'train' in path_str:
        return 'train'
    elif 'test' in path_str:
        return 'test'
    
    return 'unknown'

# Coletar imagens apenas da pasta COLOR
train_reais = []
train_fakes = []
test_reais = []
test_fakes = []
unknown_images = []

for root, dirs, files in os.walk(dataset_path):
    # IGNORAR pasta depth
    if 'depth' in root.lower():
        continue
    
    # Processar apenas pasta color
    if 'color' not in root.lower():
        continue
    
    for fname in files:
        if not fname.lower().endswith(IMG_EXTS):
            continue
        
        img_path = Path(root) / fname
        label = classify_casia_image(img_path)
        split = determine_split(img_path)
        
        if label == 'real':
            if split == 'train':
                train_reais.append(img_path)
            elif split == 'test':
                test_reais.append(img_path)
            else:
                train_reais.append(img_path)
        elif label == 'fake':
            if split == 'train':
                train_fakes.append(img_path)
            elif split == 'test':
                test_fakes.append(img_path)
            else:
                train_fakes.append(img_path)
        else:
            unknown_images.append(img_path)

print(f"\nClassificação:")
print(f"  Treino - Reais: {len(train_reais)}, Fakes: {len(train_fakes)}")
print(f"  Teste  - Reais: {len(test_reais)}, Fakes: {len(test_fakes)}")
print(f"  Não classificados: {len(unknown_images)}")

if unknown_images:
    print(f"\n[DEBUG] Exemplos não classificados:")
    for p in unknown_images[:5]:
        print(f"  {p.name}")

# 5. CRIAR VALIDAÇÃO A PARTIR DO TREINO
print("\nSeparando conjunto de validação...")
random.shuffle(train_reais)
random.shuffle(train_fakes)

n_val_reais = int(len(train_reais) * VAL_RATIO)
n_val_fakes = int(len(train_fakes) * VAL_RATIO)

val_reais = train_reais[:n_val_reais]
val_fakes = train_fakes[:n_val_fakes]
train_reais = train_reais[n_val_reais:]
train_fakes = train_fakes[n_val_fakes:]

print(f"  Validação: {len(val_reais)} reais, {len(val_fakes)} fakes")
print(f"  Treino: {len(train_reais)} reais, {len(train_fakes)} fakes")

# 6. BALANCEAMENTO DO TREINO
print("\nBalanceando conjunto de treino...")
min_train = min(len(train_reais), len(train_fakes))
if min_train > 0:
    train_reais = train_reais[:min_train]
    train_fakes = train_fakes[:min_train]
    print(f"  Treino balanceado: {min_train} reais, {min_train} fakes")

# 7. COPIAR ARQUIVOS
def copy_images(image_list, dest_dir: Path):
    """Copia imagens para o diretório de destino."""
    copied = 0
    for img_path in image_list:
        subject_id = img_path.parent.name
        new_name = f"{subject_id}_{img_path.name}"
        dest_path = dest_dir / new_name
        
        counter = 0
        while dest_path.exists():
            counter += 1
            stem = img_path.stem
            suffix = img_path.suffix
            new_name = f"{subject_id}_{stem}_{counter}{suffix}"
            dest_path = dest_dir / new_name
        
        try:
            shutil.copy2(img_path, dest_path)
            copied += 1
        except Exception as e:
            print(f"[ERRO] {img_path}: {e}")
    return copied

print("\nCopiando arquivos...")

n = copy_images(train_reais, OUTPUT_DIR / "train" / "real")
print(f"  train/real: {n} arquivos")
n = copy_images(train_fakes, OUTPUT_DIR / "train" / "fake")
print(f"  train/fake: {n} arquivos")

n = copy_images(val_reais, OUTPUT_DIR / "val" / "real")
print(f"  val/real: {n} arquivos")
n = copy_images(val_fakes, OUTPUT_DIR / "val" / "fake")
print(f"  val/fake: {n} arquivos")

n = copy_images(test_reais, OUTPUT_DIR / "test" / "real")
print(f"  test/real: {n} arquivos")
n = copy_images(test_fakes, OUTPUT_DIR / "test" / "fake")
print(f"  test/fake: {n} arquivos")

# 8. GERAR ESTATÍSTICAS E GRÁFICO
def get_folder_counts(base_dir='dataset2'):
    data = []
    for split in ['train', 'val', 'test']:
        for label in ['real', 'fake']:
            path = os.path.join(base_dir, split, label)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                data.append({'split': split, 'label': label, 'Quantidade': count})
    return pd.DataFrame(data)

print("\n" + "=" * 50)
print("RESUMO DO DATASET CASIA-FASD (dataset2) - APENAS COLOR")
print("=" * 50)

df_folders = get_folder_counts()

if not df_folders.empty and df_folders['Quantidade'].sum() > 0:
    print("\nCONTAGEM DE IMAGENS:")
    print("-" * 40)
    
    for split in ['train', 'val', 'test']:
        split_data = df_folders[df_folders['split'] == split]
        real_count = split_data[split_data['label'] == 'real']['Quantidade'].values
        fake_count = split_data[split_data['label'] == 'fake']['Quantidade'].values
        real_count = real_count[0] if len(real_count) > 0 else 0
        fake_count = fake_count[0] if len(fake_count) > 0 else 0
        print(f"  {split.upper():6s}: {real_count:5d} reais, {fake_count:5d} fakes")
    
    total = df_folders['Quantidade'].sum()
    print("-" * 40)
    print(f"  TOTAL: {total} imagens")
    
    # Gerar gráfico
    split_totals = df_folders.groupby('split')['Quantidade'].sum().sort_values().index.tolist()
    
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    ax = sns.barplot(
        data=df_folders, 
        x='split', 
        y='Quantidade', 
        hue='label', 
        order=split_totals, 
        palette={'real': 'forestgreen', 'fake': 'firebrick'}
    )

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(
                f'{int(height)}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom', fontweight='bold', 
                xytext=(0, 5), textcoords='offset points'
            )

    plt.title("CASIA-FASD (COLOR) - Quantidade de Imagens por Split", fontsize=14)
    plt.ylabel("Número de Arquivos")
    plt.xlabel("Conjunto")
    
    plt.savefig("contagem_casia_fasd.png", dpi=300, bbox_inches='tight')
    print(f"\n✓ Gráfico salvo em: contagem_casia_fasd.png")

print("\n" + "=" * 50)
print("✅ Processo concluído!")
print(f"Dataset organizado em: {OUTPUT_DIR.resolve()}")
print("=" * 50)