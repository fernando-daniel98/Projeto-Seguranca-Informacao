import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

def plotar_matriz_confusao(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ataque (0)', 'Real (1)'],
                yticklabels=['Ataque (0)', 'Real (1)'])
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Real (Ground Truth)')
    plt.title('Matriz de Confusão - Método 3')
    plt.tight_layout()
    plt.savefig('./Metodo_3/results/results_metodo3_confusion_matrix.png')
    plt.show()

def plotar_roc_curve(y_test, y_scores, eer_val):
    fpr, tpr, thresholds = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
    # Ponto EER aproximado para visualização
    plt.plot([eer_val], [1-eer_val], marker='o', markersize=5, color="red", label=f'EER (~{eer_val*100:.1f}%)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Curva ROC - Método 3')
    plt.legend(loc="lower right")
    plt.grid(linestyle='--')
    plt.savefig('./Metodo_3/results/results_metodo3_roc_curve.png')
    plt.show()

def plotar_distribuicao_scores(y_test, y_scores):
    scores_reais = y_scores[y_test == 1]
    scores_ataques = y_scores[y_test == 0]

    plt.figure(figsize=(8, 5))
    sns.histplot(scores_ataques, color="red", label="Ataques", kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(scores_reais, color="green", label="Reais", kde=True, stat="density", bins=30, alpha=0.5)
    
    plt.axvline(0, color='black', linestyle='--', label='Limiar Padrão SVM (0)')
    plt.xlabel('Score de Decisão do SVM')
    plt.ylabel('Densidade')
    plt.title('Distribuição de Scores: Reais vs. Ataques')
    plt.legend()
    plt.savefig('./Metodo_3/results/results_metodo3_score_dist.png')
    plt.show()

def calcular_hter(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn) # False Acceptance Rate
    frr = fn / (fn + tp) # False Rejection Rate
    hter = (far + frr) / 2
    return hter, far, frr

def calcular_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

print("Carregando dados de TESTE e modelo treinado...")
try:
    X_test = np.load('./Metodo_3/data/X_test_color.npy')
    y_test = np.load('./Metodo_3/data/y_test_color.npy')
    svm_clf = joblib.load('./Metodo_3/models/metodo3_svm_color.pkl')
except FileNotFoundError as e:
    print(f"Erro: Arquivo não encontrado. Verifique se rodou os passos anteriores. Detalhe: {e}")
    exit()

print(f"Dados carregados. X_test: {X_test.shape}, y_test: {y_test.shape}")

# Fazer Previsões
print("Realizando previsões no conjunto de teste...")
y_pred = svm_clf.predict(X_test)
# decision_function retorna a distância ao hiperplano, útil para EER
y_scores = svm_clf.decision_function(X_test)

# Calcular Métricas
print("\n--- RESULTADOS DO MÉTODO 3 (Cor + SVM) ---")
acc = accuracy_score(y_test, y_pred)
print(f"Acurácia: {acc*100:.2f}%")

hter, far, frr = calcular_hter(y_test, y_pred)
print(f"HTER: {hter*100:.2f}% (FAR: {far*100:.2f}%, FRR: {frr*100:.2f}%)")

eer = calcular_eer(y_test, y_scores)
print(f"EER: {eer*100:.2f}%")
print("------------------------------------------")

with open('./Metodo_3/results/results_metodo3.txt', 'w') as f:
    f.write(f"Acuracia: {acc*100:.2f}%\n")
    f.write(f"HTER: {hter*100:.2f}%\n")
    f.write(f"EER: {eer*100:.2f}%\n")


plotar_matriz_confusao(y_test, y_pred)
plotar_roc_curve(y_test, y_scores, eer)
plotar_distribuicao_scores(y_test, y_scores)