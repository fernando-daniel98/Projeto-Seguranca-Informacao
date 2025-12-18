import numpy as np
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# =============================================================================
# FUNÇÕES DE PLOTAGEM (ADAPTADAS PARA SALVAR POR SEED)
# =============================================================================

def plotar_matriz_confusao(y_test, y_pred, path_save):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Ataque (0)', 'Real (1)'],
                yticklabels=['Ataque (0)', 'Real (1)'])
    plt.xlabel('Predito pelo Modelo')
    plt.ylabel('Real (Ground Truth)')
    plt.title('Matriz de Confusão')
    plt.tight_layout()
    plt.savefig(f'{path_save}_confusion_matrix.png')
    plt.close()

def plotar_roc_curve(y_test, y_scores, eer_val, path_save):
    fpr, tpr, _ = roc_curve(y_test, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.plot([eer_val], [1-eer_val], marker='o', color="red", label=f'EER (~{eer_val*100:.1f}%)')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(linestyle='--')
    plt.savefig(f'{path_save}_roc_curve.png')
    plt.close()

def plotar_distribuicao_scores(y_test, y_scores, path_save):
    plt.figure(figsize=(8, 5))
    sns.histplot(y_scores[y_test == 0], color="red", label="Ataques", kde=True, stat="density", alpha=0.5)
    sns.histplot(y_scores[y_test == 1], color="green", label="Reais", kde=True, stat="density", alpha=0.5)
    plt.axvline(0, color='black', linestyle='--')
    plt.title('Distribuição de Scores')
    plt.legend()
    plt.savefig(f'{path_save}_score_dist.png')
    plt.close()

def calcular_hter(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    far = fp / (fp + tn)
    frr = fn / (fn + tp)
    return (far + frr) / 2, far, frr

def calcular_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer

# =============================================================================
# LOOP DE EXECUÇÃO MULTI-SEED
# =============================================================================

SEEDS = [42, 10, 23, 56, 89]
all_results = []

print("Iniciando avaliação das 5 sementes...")

for s in SEEDS:
    # 1. Carregar dados e modelo da semente atual
    data_path = f'./Metodo_3/database1/data/seed_{s}'
    model_path = f'./Metodo_3/database1/models/seed_{s}/metodo3_svm.pkl'
    results_dir = f'./Metodo_3/database1/results/seed_{s}'
    os.makedirs(results_dir, exist_ok=True)
    
    X_test = np.load(f'{data_path}/X_test.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    svm_clf = joblib.load(model_path)
    
    # 2. Previsões
    y_pred = svm_clf.predict(X_test)
    y_scores = svm_clf.decision_function(X_test)
    
    # 3. Calcular Métricas
    acc = accuracy_score(y_test, y_pred)
    hter, far, frr = calcular_hter(y_test, y_pred)
    eer = calcular_eer(y_test, y_scores)
    
    all_results.append({'seed': s, 'acc': acc, 'hter': hter, 'eer': eer})
    
    # 4. Salvar TXT individual
    with open(f'{results_dir}/results_seed_{s}.txt', 'w') as f:
        f.write(f"Seed: {s}\n")
        f.write(f"Acuracia: {acc*100:.2f}%\n")
        f.write(f"HTER: {hter*100:.2f}% (FAR: {far*100:.2f}%, FRR: {frr*100:.2f}%)\n")
        f.write(f"EER: {eer*100:.2f}%\n")
    
    # 5. Gerar Gráficos individuais
    prefixo = f'{results_dir}/seed_{s}'
    plotar_matriz_confusao(y_test, y_pred, prefixo)
    plotar_roc_curve(y_test, y_scores, eer, prefixo)
    plotar_distribuicao_scores(y_test, y_scores, prefixo)
    
    print(f"✅ Resultados da Seed {s} salvos em {results_dir}")

# =============================================================================
# RESUMO FINAL ESTATÍSTICO
# =============================================================================

df = pd.DataFrame(all_results)
print("\n" + "="*40)
print("       RESUMO ESTATÍSTICO FINAL")
print("="*40)

metricas = {
    'Acurácia': df['acc'],
    'HTER': df['hter'],
    'EER': df['eer']
}

for nome, valores in metricas.items():
    media = valores.mean() * 100
    desvio = valores.std() * 100
    print(f"{nome:8}: {media:6.2f}% (± {desvio:4.2f}%)")

print("="*40)