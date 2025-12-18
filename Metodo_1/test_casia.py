"""
test_casia.py
Testa o modelo SVM do Método 1 no dataset CASIA-FASD (dataset2).
Gera: results_metodo1_CASIA.txt, confusion_matrix_metodo1_CASIA.png, roc_curve_metodo1_CASIA.png
"""

import os
from pathlib import Path
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
import matplotlib.pyplot as plt

# Import feature extraction from the existing module
from extract_lbp import extract_lbp_features


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


def load_dataset_casia(dataset_path: Path, split: str = "test"):
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
    
    img_exts = (".jpg", ".jpeg", ".png", ".bmp")
    
    # Process Real (Label 1)
    print(f"Processando imagens 'real' de {real_dir}...")
    real_count = 0
    if real_dir.exists():
        for fname in sorted(os.listdir(real_dir)):
            if fname.lower().endswith(img_exts):
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
            if fname.lower().endswith(img_exts):
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
    
    X = np.stack(X_list)
    y = np.array(y_list)
    
    return X, y


def main():
    print("=" * 60)
    print("Teste do Método 1 (LBP + SVM) no dataset CASIA-FASD")
    print("=" * 60)
    
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho para dataset2 (CASIA-FASD)
    dataset2_path = base_dir.parent / "dataset2"
    
    if not dataset2_path.exists():
        raise SystemExit(
            f"Dataset CASIA-FASD não encontrado em {dataset2_path}. "
            "Execute database2.py primeiro."
        )
    
    # Carregar modelo treinado
    model_path = models_dir / "metodo1_lbp_svm.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"Modelo não encontrado em {model_path}. "
            "Execute svm.py primeiro para treinar o modelo."
        )
    
    print(f"\nCarregando modelo de {model_path}...")
    model = joblib.load(model_path)
    
    # Carregar dataset CASIA-FASD (test split)
    print(f"\nCarregando dataset CASIA-FASD (test) de {dataset2_path}...")
    X_test, y_test = load_dataset_casia(dataset2_path, split="test")
    
    print(f"\nDataset carregado:")
    print(f"  Total de imagens: {len(y_test)}")
    print(f"  Real (1): {np.sum(y_test == 1)}")
    print(f"  Fake (0): {np.sum(y_test == 0)}")
    print(f"  Shape de features: {X_test.shape}")
    
    # Fazer predições
    print("\nFazendo predições...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # probabilidade da classe 'real' (label=1)
    
    # Calcular métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    far, frr, hter = compute_hter(cm)
    
    # Curva ROC para estimar EER
    fpr, tpr, thresholds = roc_curve(y_test, y_proba, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
    
    auc = roc_auc_score(y_test, y_proba)
    
    report = classification_report(
        y_test, y_pred, target_names=["fake (0)", "real (1)"], zero_division=0
    )
    
    # Salvar resultados em texto
    txt_path = results_dir / "results_metodo1_CASIA.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("Método 1: LBP + SVM - Teste no CASIA-FASD (dataset2)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset: {dataset2_path.resolve()}\n")
        f.write(f"Split: test\n")
        f.write(f"Total de imagens: {len(y_test)}\n")
        f.write(f"  Real (1): {np.sum(y_test == 1)}\n")
        f.write(f"  Fake (0): {np.sum(y_test == 0)}\n\n")
        f.write("-" * 60 + "\n")
        f.write("MÉTRICAS DE DESEMPENHO\n")
        f.write("-" * 60 + "\n")
        f.write(f"Acurácia: {acc:.4f}\n")
        f.write(f"Precisão (classe real=1): {prec:.4f}\n")
        f.write(f"Recall   (classe real=1): {rec:.4f}\n\n")
        f.write("Matriz de confusão (linhas = verdade, colunas = predição)\n")
        f.write(str(cm) + "\n\n")
        f.write(f"FAR  (fake aceito como real): {far:.4f}\n")
        f.write(f"FRR  (real rejeitado como fake): {frr:.4f}\n")
        f.write(f"HTER: {hter:.4f}\n")
        f.write(f"EER (aprox.): {eer:.4f}\n\n")
        f.write(f"AUC (ROC): {auc:.4f}\n\n")
        f.write("-" * 60 + "\n")
        f.write("RELATÓRIO DE CLASSIFICAÇÃO\n")
        f.write("-" * 60 + "\n")
        f.write(report)
    
    print(f"\n✓ Resultados salvos em {txt_path}")
    print(f"\nRESUMO:")
    print(f"  Acurácia: {acc:.4f}")
    print(f"  Precisão: {prec:.4f}")
    print(f"  Recall:   {rec:.4f}")
    print(f"  HTER:     {hter:.4f}")
    print(f"  EER:      {eer:.4f}")
    print(f"  AUC:      {auc:.4f}")
    
    # ---------- Plot da matriz de confusão ----------
    fig_cm_path = results_dir / "confusion_matrix_metodo1_CASIA.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=["fake (0)", "real (1)"],
        yticklabels=["fake (0)", "real (1)"],
        ylabel="Verdadeiro",
        xlabel="Predito",
        title="Matriz de Confusão - Método 1 (LBP+SVM) - CASIA-FASD",
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
                fontsize=14,
                fontweight="bold",
            )
    
    fig.tight_layout()
    fig.savefig(fig_cm_path, dpi=300)
    plt.close(fig)
    print(f"✓ Matriz de confusão salva em {fig_cm_path}")
    
    # ---------- Plot da curva ROC ----------
    fig_roc_path = results_dir / "roc_curve_metodo1_CASIA.png"
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC (AUC={auc:.4f})")
    ax2.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random")
    ax2.set_xlabel("FPR (False Positive Rate)", fontsize=12)
    ax2.set_ylabel("TPR (True Positive Rate)", fontsize=12)
    ax2.set_title("Curva ROC - Método 1 (LBP+SVM) - CASIA-FASD", fontsize=14)
    ax2.legend(loc="lower right", fontsize=10)
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(fig_roc_path, dpi=300)
    plt.close(fig2)
    print(f"✓ Curva ROC salva em {fig_roc_path}")
    
    print("\n" + "=" * 60)
    print("✅ Teste concluído com sucesso!")
    print("=" * 60)


if __name__ == "__main__":
    main()
