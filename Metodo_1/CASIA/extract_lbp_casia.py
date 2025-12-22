"""
extract_lbp_casia.py
Extrai features LBP do dataset CASIA (dataset2) para treinar a SVM.
Salva os dados em data_casia/
"""

import os
import sys
from pathlib import Path
import numpy as np

# Adiciona o diretório pai ao path para importar extract_lbp_features
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_lbp import extract_lbp_features


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def load_casia_split(dataset_path: Path, split: str):
    """
    Carrega imagens do CASIA-FASD (dataset2) e extrai features LBP.
    
    Args:
        dataset_path: Path para dataset2/
        split: 'train', 'val' ou 'test'
    
    Returns:
        X: features LBP, shape (n_samples, n_features)
        y: labels (0=fake, 1=real), shape (n_samples,)
    """
    split_path = dataset_path / split
    if not split_path.exists():
        raise FileNotFoundError(f"Split não encontrado: {split_path}")
    
    real_dir = split_path / "real"
    fake_dir = split_path / "fake"
    
    X_list = []
    y_list = []
    
    # Process Real (Label 1)
    print(f"Processando imagens 'real' de {real_dir}...")
    real_count = 0
    if real_dir.exists():
        for fname in sorted(os.listdir(real_dir)):
            if fname.lower().endswith(IMG_EXTS):
                img_path = real_dir / fname
                try:
                    feat = extract_lbp_features(img_path)
                    X_list.append(feat)
                    y_list.append(1)
                    real_count += 1
                    if real_count % 100 == 0:
                        print(f"  Processadas {real_count} imagens reais...")
                except Exception as e:
                    print(f"  [ERRO] {fname}: {e}")
    
    print(f"  Total de imagens reais: {real_count}")
    
    # Process Fake (Label 0)
    print(f"Processando imagens 'fake' de {fake_dir}...")
    fake_count = 0
    if fake_dir.exists():
        for fname in sorted(os.listdir(fake_dir)):
            if fname.lower().endswith(IMG_EXTS):
                img_path = fake_dir / fname
                try:
                    feat = extract_lbp_features(img_path)
                    X_list.append(feat)
                    y_list.append(0)
                    fake_count += 1
                    if fake_count % 100 == 0:
                        print(f"  Processadas {fake_count} imagens fake...")
                except Exception as e:
                    print(f"  [ERRO] {fname}: {e}")
    
    print(f"  Total de imagens fake: {fake_count}")
    
    if not X_list:
        raise ValueError("Nenhuma imagem válida encontrada.")
    
    X = np.stack(X_list).astype("float32")
    y = np.array(y_list, dtype="int64")
    
    return X, y


def main():
    print("=" * 60)
    print("Extração de features LBP do dataset CASIA-FASD")
    print("=" * 60)
    
    # Diretórios
    base_dir = Path(__file__).resolve().parent
    dataset2_path = base_dir.parent.parent / "dataset2"
    output_path = base_dir / "data_casia"
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not dataset2_path.exists():
        raise SystemExit(
            f"Dataset CASIA-FASD não encontrado em {dataset2_path}. "
            "Certifique-se de que dataset2/ existe na raiz do projeto."
        )
    
    print(f"\nDataset CASIA: {dataset2_path}")
    print(f"Saída: {output_path}\n")
    
    # Extrai features para train, val e test
    print("\n=== Processando TRAIN ===")
    X_train, y_train = load_casia_split(dataset2_path, "train")
    
    print("\n=== Processando VAL ===")
    X_val, y_val = load_casia_split(dataset2_path, "val")
    
    print("\n=== Processando TEST ===")
    X_test, y_test = load_casia_split(dataset2_path, "test")
    
    # Estatísticas
    print("\n=== Resumo dos splits ===")
    print(f"Train: {len(y_train)} imagens | fake={np.sum(y_train == 0)} real={np.sum(y_train == 1)}")
    print(f"Val:   {len(y_val)} imagens | fake={np.sum(y_val == 0)} real={np.sum(y_val == 1)}")
    print(f"Test:  {len(y_test)} imagens | fake={np.sum(y_test == 0)} real={np.sum(y_test == 1)}")
    print(f"\nShape de features: {X_train.shape[1]} dimensões")
    
    # Salva arquivos
    np.save(output_path / "X_train_lbp.npy", X_train)
    np.save(output_path / "y_train_lbp.npy", y_train)
    np.save(output_path / "X_val_lbp.npy", X_val)
    np.save(output_path / "y_val_lbp.npy", y_val)
    np.save(output_path / "X_test_lbp.npy", X_test)
    np.save(output_path / "y_test_lbp.npy", y_test)
    
    print(f"\n✓ Features LBP salvas em {output_path}")


if __name__ == "__main__":
    main()
