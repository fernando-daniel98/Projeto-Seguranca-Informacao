"""
test_nuaa.py
Testa o modelo SVM treinado no CASIA no dataset NUAA.
Gera resultados em results_casia/
"""

import os
import sys
from pathlib import Path
import numpy as np
import joblib
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
import matplotlib.pyplot as plt

# Adiciona o diretório pai ao path para importar extract_lbp_features
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from extract_lbp import extract_lbp_features


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


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


def load_nuaa_split(dataset_path: Path, split: str):
    """
    Carrega imagens do NUAA e extrai features LBP.
    
    Args:
        dataset_path: Path para dataset/
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
    
    # Process Real (Label 1)
    print(f"Processando imagens 'real' de {real_dir}...")
    real_count = 0
    if real_dir.exists():
        for fname in sorted(os.listdir(real_dir)):
            if fname.lower().endswith(IMG_EXTS):
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
            if fname.lower().endswith(IMG_EXTS):
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
    
    X = np.stack(X_list).astype("float32")
    y = np.array(y_list, dtype="int64")
    
    return X, y


def main():
    print("=" * 60)
    print("Teste do Modelo SVM (treinado no CASIA) no dataset NUAA")
    print("=" * 60)
    
    # Diretórios
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models_casia"
    results_dir = base_dir / "results_casia"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Caminho para dataset NUAA (dataset/)
    dataset_nuaa_path = base_dir.parent.parent / "dataset"
    
    if not dataset_nuaa_path.exists():
        raise SystemExit(
            f"Dataset NUAA não encontrado em {dataset_nuaa_path}. "
            "Certifique-se de que dataset/ existe na raiz do projeto."
        )
    
    # Carregar modelo treinado
    model_path = models_dir / "svm_casia_trained.pkl"
    if not model_path.exists():
        raise SystemExit(
            f"Modelo não encontrado em {model_path}. "
            "Execute train_svm_casia.py primeiro."
        )
    
    print(f"\nCarregando modelo de {model_path}...")
    model = joblib.load(model_path)
    
    # Carregar dataset NUAA (test split)
    print(f"\nCarregando dataset NUAA (test) de {dataset_nuaa_path}...")
    X_test, y_test = load_nuaa_split(dataset_nuaa_path, split="test")
    
    print(f"\nDataset carregado:")
    print(f"  Total de imagens: {len(y_test)}")
    print(f"  Real (1): {np.sum(y_test == 1)}")
    print(f"  Fake (0): {np.sum(y_test == 0)}")
    print(f"  Shape de features: {X_test.shape}")
    
    # Predições
    print("\n=== Realizando predições ===")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba)
    
    cm = confusion_matrix(y_test, y_pred)
    far, frr, hter = compute_hter(cm)
    
    print("\n=== Resultados ===")
    print(f"Acurácia:  {acc:.4f}")
    print(f"Precisão:  {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    print(f"FAR:       {far:.4f}")
    print(f"FRR:       {frr:.4f}")
    print(f"HTER:      {hter:.4f}")
    
    print("\nMatriz de Confusão:")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    
    # Salvar resultados em arquivo de texto
    txt_path = results_dir / "results_casia_to_nuaa.txt"
    with open(txt_path, "w") as f:
        f.write("Método 1: LBP + SVM\n")
        f.write("Treinamento: CASIA-FASD (dataset2)\n")
        f.write("Teste: NUAA (dataset)\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Dataset de teste (NUAA):\n")
        f.write(f"  Total de imagens: {len(y_test)}\n")
        f.write(f"  Real (1): {np.sum(y_test == 1)}\n")
        f.write(f"  Fake (0): {np.sum(y_test == 0)}\n\n")
        
        f.write("Resultados:\n")
        f.write(f"  Acurácia:  {acc:.4f}\n")
        f.write(f"  Precisão:  {prec:.4f}\n")
        f.write(f"  Recall:    {rec:.4f}\n")
        f.write(f"  F1-Score:  {f1:.4f}\n")
        f.write(f"  AUC-ROC:   {auc:.4f}\n")
        f.write(f"  FAR:       {far:.4f}\n")
        f.write(f"  FRR:       {frr:.4f}\n")
        f.write(f"  HTER:      {hter:.4f}\n\n")
        
        f.write("Matriz de Confusão:\n")
        f.write(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],\n")
        f.write(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]\n\n")
        
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))
    
    print(f"\n✓ Resultados salvos em {txt_path}")
    
    # Plotar Matriz de Confusão
    fig_cm, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    classes = ["Fake", "Real"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title="Matriz de Confusão - SVM (CASIA→NUAA)",
        ylabel="Classe Verdadeira",
        xlabel="Classe Predita"
    )
    
    # Adicionar valores nas células
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=14)
    
    fig_cm.tight_layout()
    fig_cm_path = results_dir / "confusion_matrix_casia_to_nuaa.png"
    fig_cm.savefig(fig_cm_path, dpi=150)
    print(f"✓ Matriz de confusão salva em {fig_cm_path}")
    
    # Plotar Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    fig_roc, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.4f})')
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Curva ROC - SVM (CASIA→NUAA)', fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    fig_roc.tight_layout()
    fig_roc_path = results_dir / "roc_curve_casia_to_nuaa.png"
    fig_roc.savefig(fig_roc_path, dpi=150)
    print(f"✓ Curva ROC salva em {fig_roc_path}")
    
    print("\n" + "=" * 60)
    print("Teste concluído!")
    print("=" * 60)


if __name__ == "__main__":
    main()
