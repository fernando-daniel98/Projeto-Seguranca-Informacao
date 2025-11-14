from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib


def main():
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Carrega treino e validação separadamente
    X_train = np.load(data_dir / "X_train_lbp.npy")
    y_train = np.load(data_dir / "y_train_lbp.npy")
    X_val = np.load(data_dir / "X_val_lbp.npy")
    y_val = np.load(data_dir / "y_val_lbp.npy")

    print("Shapes:")
    print("  Train:", X_train.shape, y_train.shape)
    print("  Val:  ", X_val.shape, y_val.shape)

    # Grade de hiperparâmetros para C e gamma
    param_grid = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", 0.01, 0.001],
    }

    print("=== GridSearch on TRAIN only, validate on VAL (no data leakage) ===")
    
    best_score = 0
    best_params = None
    best_pipeline = None
    
    # Manual grid search to avoid data leakage
    for C in param_grid["svm__C"]:
        for gamma in param_grid["svm__gamma"]:
            # Create fresh pipeline for each combination
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(kernel="rbf", C=C, gamma=gamma, 
                           class_weight="balanced", probability=True))
            ])
            
            # Fit ONLY on training data
            pipe.fit(X_train, y_train)
            
            # Validate on validation set
            val_score = pipe.score(X_val, y_val)
            
            print(f"C={C}, gamma={gamma}: val_acc={val_score:.4f}")
            
            if val_score > best_score:
                best_score = val_score
                best_params = {"svm__C": C, "svm__gamma": gamma}
                best_pipeline = pipe

    print(f"\nMelhores hiperparâmetros: {best_params}")
    print(f"Melhor acurácia na validação: {best_score:.4f}")

    # Avalia explicitamente na parte de validação
    y_val_pred = best_pipeline.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"\nAcurácia em VAL: {val_acc:.4f}")
    print("Relatório em validação:")
    print(classification_report(y_val, y_val_pred, target_names=["fake (0)", "real (1)"]))

    # Retrain on train+val with best params for final model
    print("\n=== Treinando modelo final em TRAIN+VAL ===")
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", **{k.replace("svm__", ""): v for k, v in best_params.items()},
                   class_weight="balanced", probability=True))
    ])
    
    X_combined = np.vstack([X_train, X_val])
    y_combined = np.concatenate([y_train, y_val])
    final_pipeline.fit(X_combined, y_combined)

    # === EVALUATE ON TEST SET ===
    X_test = np.load(data_dir / "X_test_lbp.npy")
    y_test = np.load(data_dir / "y_test_lbp.npy")
    
    print(f"\n{'='*50}")
    print("AVALIAÇÃO FINAL NO CONJUNTO DE TESTE")
    print(f"{'='*50}")
    
    y_test_pred = final_pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    print(f"Test Set Accuracy: {test_acc:.4f}")
    print("\nTest Set Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["fake (0)", "real (1)"]))

    model_path = models_dir / "metodo1_lbp_svm.pkl"
    joblib.dump(final_pipeline, model_path)
    print(f"\nModelo final (treinado em train+val) salvo em {model_path}")


if __name__ == "__main__":
    main()
