import os
import cv2
import numpy as np

# Configuração das 5 seeds conforme database.py
SEEDS = [42, 10, 23, 56, 89]
categorias = {'real': 1, 'fake': 0}

def extrair_caracteristicas_cor(caminho_imagem, bins=32):
    img = cv2.imread(caminho_imagem)
    if img is None: return None
    img = cv2.resize(img, (224, 224))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    features = []
    for img_src, channel_idx in [(hsv, 0), (hsv, 1), (ycbcr, 1), (ycbcr, 2)]:
        hist = cv2.calcHist([img_src], [channel_idx], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features.extend(hist.flatten())
    return np.array(features)

def processar_diretorio(diretorio_base):
    X, y = [], []
    for cat_nome, label in categorias.items():
        caminho = os.path.join(diretorio_base, cat_nome)
        if not os.path.isdir(caminho): continue
        for arq in os.listdir(caminho):
            img_path = os.path.join(caminho, arq)
            feat = extrair_caracteristicas_cor(img_path)
            if feat is not None:
                X.append(feat)
                y.append(label)
    return np.array(X), np.array(y)

for s in SEEDS:
    print(f"\n--- Extraindo Features para SEED {s} ---")
    base_dir = f"./dataset_seed_{s}"
    X_train, y_train = processar_diretorio(os.path.join(base_dir, 'train'))
    X_test, y_test = processar_diretorio(os.path.join(base_dir, 'test'))
    
    out_path = f'./Metodo_3/data/seed_{s}'
    os.makedirs(out_path, exist_ok=True)
    np.save(f'{out_path}/X_train.npy', X_train)
    np.save(f'{out_path}/y_train.npy', y_train)
    np.save(f'{out_path}/X_test.npy', X_test)
    np.save(f'{out_path}/y_test.npy', y_test)