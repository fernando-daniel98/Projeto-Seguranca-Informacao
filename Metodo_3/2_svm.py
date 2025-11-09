import numpy as np
from sklearn.svm import SVC
import joblib
import os

# Carregar os dados de TREINO salvos no passo anterior
print("Carregando dados de treino...")
try:
    X_train = np.load('./Metodo_3/data/X_train_color.npy')
    y_train = np.load('./Metodo_3/data/y_train_color.npy')
except FileNotFoundError:
    print("Erro: Arquivos de features não encontrados. Rode o script de extração primeiro.")
    exit()

print(f"Dados carregados. X_train: {X_train.shape}, y_train: {y_train.shape}")

# Configurar e treinar o SVM
print("Iniciando treinamento do SVM (RBF)...")
svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)
print("Treinamento concluído.")

# Salvar o modelo treinado
os.makedirs('./Metodo_3/models', exist_ok=True)
caminho_modelo = './Metodo_3/models/metodo3_svm_color.pkl'
joblib.dump(svm_clf, caminho_modelo)
print(f"Modelo treinado salvo em: {caminho_modelo}")