"""
train_svm_casia.py
Treina a SVM com features LBP extraídas do dataset CASIA.
Salva o modelo em models_casia/
"""

import sys
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib


def main():
    print("=" * 60)
    print("Treinamento da SVM com dataset CASIA-FASD")
    print("=" * 60)
    
    # Diretórios
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data_casia"
    models_dir = base_dir / "models_casia"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Verifica se as features foram extraídas
    if not data_dir.exists():
        raise SystemExit(
            f"Diretório de dados não encontrado: {data_dir}\n"
            "Execute extract_lbp_casia.py primeiro."
        )
    
    # Carrega dados de treino e validação
    print("\nCarregando features LBP do CASIA...")
    X_train = np.load(data_dir / "X_train_lbp.npy")
    y_train = np.load(data_dir / "y_train_lbp.npy")
    X_val = np.load(data_dir / "X_val_lbp.npy")
    y_val = np.load(data_dir / "y_val_lbp.npy")
    
    print("\nShapes:")
    print(f"  Train: {X_train.shape} | Labels: {y_train.shape}")
    print(f"  Val:   {X_val.shape} | Labels: {y_val.shape}")
    
    print("\nDistribuição de classes (Train):")
    print(f"  Fake (0): {np.sum(y_train == 0)}")
    print(f"  Real (1): {np.sum(y_train == 1)}")
    
    # Pipeline: StandardScaler + SVM
    print("\n=== Treinando SVM com GridSearchCV ===")
    
    # Grade de hiperparâmetros
    param_grid = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }
    
    base_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", class_weight="balanced", probability=True)),
    ])
    
    # GridSearchCV com validação no conjunto de validação
    # Combinamos train+val para o grid search
    X_trainval = np.vstack([X_train, X_val])
    y_trainval = np.concatenate([y_train, y_val])
    
    # Criamos índices de split manual (train vs val)
    train_indices = np.arange(len(X_train))
    val_indices = np.arange(len(X_train), len(X_trainval))
    cv_split = [(train_indices, val_indices)]
    
    grid = GridSearchCV(
        base_pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv_split,
        refit=False,
        n_jobs=-1,
        verbose=2,
    )
    
    print("\nIniciando GridSearchCV...")
    grid.fit(X_trainval, y_trainval)
    
    print("\n=== Melhores hiperparâmetros ===")
    print(f"C: {grid.best_params_['svm__C']}")
    print(f"gamma: {grid.best_params_['svm__gamma']}")
    print(f"Melhor AUC (validação): {grid.best_score_:.4f}")
    
    # Avaliação "limpa": treina o melhor modelo APENAS no treino e mede no val.
    # Isso evita vazamento quando GridSearchCV usa refit=True e refaz treino em train+val.
    print("\n=== Avaliação no conjunto de validação (limpa) ===")
    best_val_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            C=grid.best_params_["svm__C"],
            gamma=grid.best_params_["svm__gamma"],
        )),
    ])
    best_val_model.fit(X_train, y_train)
    y_val_pred = best_val_model.predict(X_val)
    y_val_proba = best_val_model.predict_proba(X_val)[:, 1]
    
    val_acc = accuracy_score(y_val, y_val_pred)
    val_auc = roc_auc_score(y_val, y_val_proba)
    
    print(f"Acurácia: {val_acc:.4f}")
    print(f"AUC-ROC:  {val_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred, target_names=["Fake", "Real"]))
    
    # Retreina no conjunto completo (train+val) com os melhores parâmetros
    print("\n=== Retreinando no conjunto completo (train+val) ===")
    best_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            C=grid.best_params_["svm__C"],
            gamma=grid.best_params_["svm__gamma"],
        )),
    ])
    
    best_model.fit(X_trainval, y_trainval)
    
    # Salva o modelo
    model_path = models_dir / "svm_casia_trained.pkl"
    joblib.dump(best_model, model_path)
    print(f"\n✓ Modelo salvo em {model_path}")
    
    # Salva também os melhores parâmetros
    params_path = models_dir / "best_params.txt"
    with open(params_path, "w") as f:
        f.write(f"Melhores Hiperparâmetros:\n")
        f.write(f"C: {grid.best_params_['svm__C']}\n")
        f.write(f"gamma: {grid.best_params_['svm__gamma']}\n")
        f.write(f"\nMelhor AUC (validação): {grid.best_score_:.4f}\n")
        f.write(f"\nAvaliação no conjunto de validação:\n")
        f.write(f"Acurácia: {val_acc:.4f}\n")
        f.write(f"AUC-ROC:  {val_auc:.4f}\n")
    
    print(f"✓ Parâmetros salvos em {params_path}")
    print("\nTreinamento concluído!")


if __name__ == "__main__":
    main()
