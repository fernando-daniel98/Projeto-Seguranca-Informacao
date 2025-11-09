import os
import cv2
import numpy as np

def extrair_caracteristicas_cor(caminho_imagem, bins=32):
    """
    Implementa a extração de características do Método 3.
    Converte para HSV e YCbCr e extrai histogramas dos canais H, S, Cb, Cr.
    """
    img = cv2.imread(caminho_imagem)
    if img is None:
        return None

    # Redimensionamento para padronização 
    img = cv2.resize(img, (224, 224))

    # Conversão de Espaço de Cor 
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Extração de Histogramas dos canais relevantes 
    # HSV: Canal 0 (Hue), Canal 1 (Saturation)
    # YCrCb: Canal 1 (Cr), Canal 2 (Cb) - OpenCV usa ordem Y-Cr-Cb
    
    features = []
    
    # Calcula histograma para cada canal desejado, normaliza e adiciona à lista
    # Usamos 32 bins por canal
    for img_src, channel_idx in [(hsv, 0), (hsv, 1), (ycbcr, 1), (ycbcr, 2)]:
        hist = cv2.calcHist([img_src], [channel_idx], None, [bins], [0, 256])
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
        features.extend(hist.flatten())

    # Vetor de Características Concatenado
    return np.array(features)

def processar_diretorio(diretorio_base, categorias):
    X = []
    y = []
    
    for categoria_nome, label in categorias.items():
        caminho_categoria = os.path.join(diretorio_base, categoria_nome)
        if not os.path.isdir(caminho_categoria):
            print(f"Aviso: Diretório não encontrado: {caminho_categoria}")
            continue

        print(f"Processando {categoria_nome} em {caminho_categoria}...")
        for nome_arquivo in os.listdir(caminho_categoria):
            caminho_imagem = os.path.join(caminho_categoria, nome_arquivo)
            # Verifica se é um arquivo de imagem válido
            if os.path.isfile(caminho_imagem) and nome_arquivo.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                 feat = extrair_caracteristicas_cor(caminho_imagem)
                 if feat is not None:
                     X.append(feat)
                     y.append(label)
    return np.array(X), np.array(y)

base_dir_treino = './dataset/train' 
base_dir_teste = './dataset/test'   
categorias = {'real': 1, 'fake': 0} 

# Execução do Processamento
print("Iniciando extração de características para TREINO...")
X_train, y_train = processar_diretorio(base_dir_treino, categorias)
print(f"Treino concluído. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

print("\nIniciando extração de características para TESTE...")
X_test, y_test = processar_diretorio(base_dir_teste, categorias)
print(f"Teste concluído. X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

os.makedirs('./Metodo_3/data', exist_ok=True)
np.save('./Metodo_3/data/X_train_color.npy', X_train)
np.save('./Metodo_3/data/y_train_color.npy', y_train)
np.save('./Metodo_3/data/X_test_color.npy', X_test)
np.save('./Metodo_3/data/y_test_color.npy', y_test)
print("\nCaracterísticas extraídas e salvas em ./Metodo_3/data/")