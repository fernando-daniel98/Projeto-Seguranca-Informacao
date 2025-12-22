"""
Gera métricas detalhadas e visualizações para o Método 2 (VGG16).
Utiliza o modelo treinado salvo no diretório models/.
"""

import numpy as np
import tensorflow as tf
from keras import models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def compute_hter(cm: np.ndarray):
    """
    Calcula FAR, FRR e HTER a partir da matriz de confusão.
    cm:
        [[TN, FP],
         [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-12)  # False Acceptance Rate
    frr = fn / (fn + tp + 1e-12)  # False Rejection Rate
    hter = (far + frr) / 2.0
    return far, frr, hter


def compute_eer(y_true, y_scores):
    """Calcula o Equal Error Rate (EER)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        # Fallback: aproximação pelo ponto mais próximo
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer


def plot_confusion_matrix(cm, save_path):
    """Plota e salva a matriz de confusão."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        cbar=True,
        xticklabels=['Fake (0)', 'Real (1)'],
        yticklabels=['Fake (0)', 'Real (1)'],
        annot_kws={"size": 14}
    )
    plt.xlabel('Predito pelo Modelo', fontsize=12)
    plt.ylabel('Real (Ground Truth)', fontsize=12)
    plt.title('Matriz de Confusão - Método 2 (VGG16)', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matriz de confusão salva em: {save_path}")


def plot_roc_curve(y_true, y_scores, eer_val, auc_val, save_path):
    """Plota e salva a curva ROC."""
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
    
    # Ponto EER
    plt.plot([eer_val], [1-eer_val], marker='o', markersize=8, color="red", 
             label=f'EER ({eer_val*100:.2f}%)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Curva ROC - Método 2 (VGG16)', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curva ROC salva em: {save_path}")


def plot_score_distribution(y_true, y_scores, save_path):
    """Plota a distribuição dos scores para classes real e fake."""
    scores_reais = y_scores[y_true == 1]
    scores_fakes = y_scores[y_true == 0]

    plt.figure(figsize=(10, 6))
    sns.histplot(scores_fakes, color="red", label="Fake (Ataque)", 
                 kde=True, stat="density", bins=30, alpha=0.5)
    sns.histplot(scores_reais, color="green", label="Real", 
                 kde=True, stat="density", bins=30, alpha=0.5)
    
    plt.axvline(0.5, color='black', linestyle='--', label='Limiar Padrão (0.5)')
    plt.xlabel('Probabilidade Predita (Classe Real)', fontsize=12)
    plt.ylabel('Densidade', fontsize=12)
    plt.title('Distribuição de Scores: Reais vs. Fakes - Método 2', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Distribuição de scores salva em: {save_path}")


def main():
    # Configurar diretórios
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Caminho do dataset de teste (CASIA-FASD)
    test_dir = PROJECT_ROOT / "dataset2" / "test"

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Verificar se o modelo existe
    model_path = MODELS_DIR / "metodo2_vgg16_best.keras"
    if not model_path.exists():
        # Tentar modelo final
        model_path = MODELS_DIR / "metodo2_vgg16_final.keras"
        if not model_path.exists():
            raise SystemExit(
                f"Modelo não encontrado em {MODELS_DIR}. "
                "Rode primeiro: python Metodo_2/main.py"
            )

    print(f"Carregando modelo de: {model_path}")
    model = models.load_model(str(model_path))

    # Carregar dataset de teste
    print(f"Carregando dataset de teste de: {test_dir}")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    # Extrair labels verdadeiros e fazer predições
    y_true = []
    y_pred_proba = []

    print("Realizando predições...")
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions[:, 1])  # Probabilidade da classe "real"

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # --- CÁLCULO DAS MÉTRICAS ---
    print("\n" + "="*50)
    print("RESULTADOS DO MÉTODO 2 (VGG16 Transfer Learning)")
    print("Treinado e Testado em: CASIA-FASD (dataset2)")
    print("="*50)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    cm = confusion_matrix(y_true, y_pred)

    far, frr, hter = compute_hter(cm)
    eer = compute_eer(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)

    report = classification_report(
        y_true, y_pred, target_names=["Fake (0)", "Real (1)"]
    )

    # Exibir resultados
    print(f"\nAcurácia:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nFAR (False Acceptance Rate): {far:.4f} ({far*100:.2f}%)")
    print(f"FRR (False Rejection Rate):  {frr:.4f} ({frr*100:.2f}%)")
    print(f"HTER: {hter:.4f} ({hter*100:.2f}%)")
    print(f"EER:  {eer:.4f} ({eer*100:.2f}%)")
    print(f"AUC:  {auc_score:.4f}")
    print(f"\nMatriz de Confusão:")
    print(cm)
    print(f"\nRelatório de Classificação:\n{report}")

    # --- SALVAR RESULTADOS EM ARQUIVO ---
    txt_path = RESULTS_DIR / "results_metodo2.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("="*50 + "\n")
        f.write("RESULTADOS DO MÉTODO 2 (VGG16 Transfer Learning)\n")
        f.write("Treinado e Testado em: CASIA-FASD (dataset2)\n")
        f.write("="*50 + "\n\n")
        f.write(f"Acurácia:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precisão:  {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write(f"FAR (False Acceptance Rate): {far:.4f} ({far*100:.2f}%)\n")
        f.write(f"FRR (False Rejection Rate):  {frr:.4f} ({frr*100:.2f}%)\n")
        f.write(f"HTER: {hter:.4f} ({hter*100:.2f}%)\n")
        f.write(f"EER:  {eer:.4f} ({eer*100:.2f}%)\n")
        f.write(f"AUC:  {auc_score:.4f}\n\n")
        f.write("Matriz de Confusão (linhas = verdade, colunas = predição):\n")
        f.write(str(cm) + "\n\n")
        f.write("Relatório de Classificação:\n")
        f.write(report)

    print(f"\nResultados salvos em: {txt_path}")

    # --- GERAR VISUALIZAÇÕES ---
    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix_metodo2.png")
    plot_roc_curve(y_true, y_pred_proba, eer, auc_score, RESULTS_DIR / "roc_curve_metodo2.png")
    plot_score_distribution(y_true, y_pred_proba, RESULTS_DIR / "score_distribution_metodo2.png")

    print("\n✅ Todas as métricas e visualizações foram geradas com sucesso!")


if __name__ == "__main__":
    main()