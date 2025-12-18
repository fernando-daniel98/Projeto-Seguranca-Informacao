"""
eval_dataset2.py
Avalia o modelo treinado no NUAA usando o dataset2 (CASIA-FASD).
Testa a generalização cross-dataset do modelo.
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
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-12)
    frr = fn / (fn + tp + 1e-12)
    hter = (far + frr) / 2.0
    return far, frr, hter


def compute_eer(y_true, y_scores):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    try:
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except ValueError:
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.abs(fnr - fpr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    return eer


def plot_confusion_matrix(cm, save_path, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Oranges', cbar=True,
        xticklabels=['Fake (0)', 'Real (1)'],
        yticklabels=['Fake (0)', 'Real (1)'],
        annot_kws={"size": 14}
    )
    plt.xlabel('Predito pelo Modelo', fontsize=12)
    plt.ylabel('Real (Ground Truth)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_scores, eer, auc_val, save_path, title):
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_val:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório')
    plt.plot([eer], [1-eer], marker='o', markersize=8, color="red", label=f'EER ({eer*100:.2f}%)')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    BASE_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = BASE_DIR.parent
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Dataset2 (CASIA-FASD)
    dataset2_test = PROJECT_ROOT / "dataset2" / "test"
    
    if not dataset2_test.exists():
        raise SystemExit(
            f"❌ Dataset2 não encontrado em {dataset2_test}\n"
            "   Rode primeiro: python database2.py"
        )

    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32

    # Carregar modelo
    model_path = MODELS_DIR / "metodo2_vgg16_best.keras"
    if not model_path.exists():
        model_path = MODELS_DIR / "metodo2_vgg16_final.keras"
        if not model_path.exists():
            raise SystemExit(
                f"❌ Modelo não encontrado em {MODELS_DIR}\n"
                "   Rode primeiro: python -m Metodo_2.main"
            )

    print("=" * 60)
    print("AVALIAÇÃO CROSS-DATASET")
    print("=" * 60)
    print(f"Modelo treinado em: NUAA (dataset)")
    print(f"Avaliando em: CASIA-FASD (dataset2)")
    print(f"\nCarregando modelo: {model_path.name}")
    
    model = models.load_model(str(model_path))

    print(f"Carregando dataset2/test...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        str(dataset2_test),
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        shuffle=False
    )

    # Predições
    y_true = []
    y_pred_proba = []

    print("Realizando predições...")
    for images, labels in test_ds:
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        predictions = model.predict(images, verbose=0)
        y_pred_proba.extend(predictions[:, 1])

    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Métricas
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    far, frr, hter = compute_hter(cm)
    
    try:
        eer = compute_eer(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
    except:
        eer, auc_score = 0.0, 0.0

    report = classification_report(y_true, y_pred, target_names=["Fake (0)", "Real (1)"], zero_division=0)

    # Exibir
    print("\n" + "=" * 60)
    print("RESULTADOS: NUAA → CASIA-FASD")
    print("=" * 60)
    print(f"\nAcurácia:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"\nFAR:  {far:.4f} ({far*100:.2f}%)")
    print(f"FRR:  {frr:.4f} ({frr*100:.2f}%)")
    print(f"HTER: {hter:.4f} ({hter*100:.2f}%)")
    print(f"EER:  {eer:.4f} ({eer*100:.2f}%)")
    print(f"AUC:  {auc_score:.4f}")
    print(f"\nMatriz de Confusão:\n{cm}")
    print(f"\n{report}")

    # Salvar resultados
    txt_path = RESULTS_DIR / "results_cross_dataset2.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("AVALIAÇÃO CROSS-DATASET: NUAA → CASIA-FASD\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Acurácia:  {acc:.4f} ({acc*100:.2f}%)\n")
        f.write(f"Precisão:  {prec:.4f}\n")
        f.write(f"Recall:    {rec:.4f}\n")
        f.write(f"F1-Score:  {f1:.4f}\n\n")
        f.write(f"FAR:  {far:.4f} ({far*100:.2f}%)\n")
        f.write(f"FRR:  {frr:.4f} ({frr*100:.2f}%)\n")
        f.write(f"HTER: {hter:.4f} ({hter*100:.2f}%)\n")
        f.write(f"EER:  {eer:.4f} ({eer*100:.2f}%)\n")
        f.write(f"AUC:  {auc_score:.4f}\n\n")
        f.write("Matriz de Confusão:\n")
        f.write(str(cm) + "\n\n")
        f.write("Relatório:\n")
        f.write(report)

    print(f"\n✓ Resultados salvos em: {txt_path}")

    # Gráficos
    plot_confusion_matrix(cm, RESULTS_DIR / "confusion_matrix_dataset2.png", 
                          "Cross-Dataset: NUAA → CASIA-FASD")
    print(f"✓ Matriz de confusão salva")
    
    plot_roc_curve(y_true, y_pred_proba, eer, auc_score, 
                   RESULTS_DIR / "roc_curve_dataset2.png",
                   "Curva ROC - Cross-Dataset (NUAA → CASIA-FASD)")
    print(f"✓ Curva ROC salva")

    print("\n" + "=" * 60)
    print("✅ Avaliação cross-dataset concluída!")
    print("=" * 60)


if __name__ == "__main__":
    main()