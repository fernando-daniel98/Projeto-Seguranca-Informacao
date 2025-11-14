from pathlib import Path

import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
import matplotlib.pyplot as plt


def compute_hter(cm: np.ndarray):
    """
    Calcula FAR, FRR e HTER a partir da matriz de confusão.

    cm:
        [[TN, FP],
         [FN, TP]]
    """
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn + 1e-12)  # False Acceptance Rate (fake -> real)
    frr = fn / (fn + tp + 1e-12)  # False Rejection Rate (real -> fake)
    hter = (far + frr) / 2.0
    return far, frr, hter


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_test = np.load(data_dir / "X_test_lbp.npy")
    y_test = np.load(data_dir / "y_test_lbp.npy")

    model_path = models_dir / "metodo1_lbp_svm.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"Modelo não encontrado em {model_path}. "
            "Rode primeiro: Metodo_1/2_svm.py"
        )

    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    # probabilidade da classe 'real' (label=1)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)

    far, frr, hter = compute_hter(cm)

    # Curva ROC para estimar EER
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0

    report = classification_report(
        y_test, y_pred, target_names=["fake (0)", "real (1)"]
    )

    txt_path = results_dir / "results_metodo1.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Método 1: LBP + SVM ===\n\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão (classe real=1): {prec:.4f}\n")
        f.write(f"Recall   (classe real=1): {rec:.4f}\n\n")
        f.write("Matriz de confusão (linhas = verdade, colunas = predição)\n")
        f.write(str(cm) + "\n\n")
        f.write(f"FAR  (fake aceito como real): {far:.4f}\n")
        f.write(f"FRR  (real rejeitado como fake): {frr:.4f}\n")
        f.write(f"HTER: {hter:.4f}\n")
        f.write(f"EER (aprox.): {eer:.4f}\n\n")
        f.write("Relatório de classificação:\n")
        f.write(report)

    print(f"Resultados salvos em {txt_path}")
    print(
        f"Acurácia: {acc:.4f} | Precisão: {prec:.4f} | "
        f"Recall: {rec:.4f} | HTER: {hter:.4f} | EER ~ {eer:.4f}"
    )

    # ---------- Plot da matriz de confusão ----------
    fig_cm_path = results_dir / "confusion_matrix_metodo1.png"
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["fake (0)", "real (1)"],
        yticklabels=["fake (0)", "real (1)"],
        ylabel="Verdadeiro",
        xlabel="Predito",
        title="Matriz de Confusão - Método 1 (LBP+SVM)",
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(fig_cm_path)
    plt.close(fig)
    print(f"Matriz de confusão salva em {fig_cm_path}")

    # ---------- Plot da curva ROC ----------
    fig_roc_path = results_dir / "roc_curve_metodo1.png"
    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr, label="ROC (AUC não calculada)")
    ax2.plot([0, 1], [0, 1], "--")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.set_title("Curva ROC - Método 1 (LBP+SVM)")
    ax2.legend(loc="lower right")
    fig2.tight_layout()
    fig2.savefig(fig_roc_path)
    plt.close(fig2)
    print(f"Curva ROC salva em {fig_roc_path}")


if __name__ == "__main__":
    main()