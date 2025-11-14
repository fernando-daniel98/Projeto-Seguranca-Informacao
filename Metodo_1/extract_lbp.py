from pathlib import Path
import os

import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp_features(
    image_path,
    radius=2,
    n_points=None,
    method="uniform",
    resize_to=(224, 224),
):
    """
    Lê uma imagem, converte para cinza, redimensiona e extrai
    histograma LBP normalizado.
    """
    if n_points is None:
        n_points = 8 * radius

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if resize_to is not None:
        gray = cv2.resize(gray, resize_to)

    lbp = local_binary_pattern(gray, n_points, radius, method)

    n_bins = int(lbp.max() + 1)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True,  # histograma normalizado
    )

    return hist.astype("float32")

def load_split_features(dataset_dir: Path, split: str):
    """
    Carrega todas as imagens de dataset/<split>/{real,fake}
    e retorna X (features LBP) e y (labels 0/1).
    """
    split_dir = dataset_dir / split
    features = []
    labels = []

    # Mapeamento de labels conforme README
    label_map = {"real": 1, "fake": 0}
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")

    for label_name, label_value in label_map.items():
        class_dir = split_dir / label_name
        if not class_dir.exists():
            print(f"[AVISO] Diretório não encontrado: {class_dir}")
            continue

        for root, _, files in os.walk(class_dir):
            for fname in files:
                if not fname.lower().endswith(img_exts):
                    continue

                img_path = Path(root) / fname
                try:
                    feat = extract_lbp_features(img_path)
                    features.append(feat)
                    labels.append(label_value)
                except Exception as e:
                    print(f"[ERRO] {img_path}: {e}")

    if not features:
        raise RuntimeError(f"Nenhuma imagem encontrada em {split_dir}")

    X = np.stack(features).astype("float32")
    y = np.array(labels, dtype="int64")
    return X, y

def main():
    base_dir = Path(__file__).resolve().parent
    print(f"Base directory: {base_dir}")
    dataset_path = base_dir / "dataset"
    print(f"Dataset path: {dataset_path}")
    output_path = base_dir / "data"
    print(f"Output path: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)

    print("=== Extraindo LBP para o conjunto de TREINO ===")
    X_train, y_train = load_split_features(dataset_path, "train")
    print(f"Train: {X_train.shape}, labels: {np.bincount(y_train)}")

    print("=== Extraindo LBP para o conjunto de VALIDAÇÃO ===")
    X_val, y_val = load_split_features(dataset_path, "val")
    print(f"Val:   {X_val.shape}, labels: {np.bincount(y_val)}")

    print("=== Extraindo LBP para o conjunto de TESTE ===")
    X_test, y_test = load_split_features(dataset_path, "test")
    print(f"Test:  {X_test.shape}, labels: {np.bincount(y_test)}")

    np.save(output_path / "X_train_lbp.npy", X_train)
    np.save(output_path / "y_train_lbp.npy", y_train)
    np.save(output_path / "X_val_lbp.npy", X_val)
    np.save(output_path / "y_val_lbp.npy", y_val)
    np.save(output_path / "X_test_lbp.npy", X_test)
    np.save(output_path / "y_test_lbp.npy", y_test)

    print(f"Features LBP salvas em {output_path}")

if __name__ == "__main__":
    main()